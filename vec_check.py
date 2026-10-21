#!/usr/bin/env python3
"""vec_check: offline format check of a VEC prediction .h5ad against a board contract.

Downloads (and caches under ~/.cache/vec-kit) the public board contracts from
https://virtualembryo.ai/challenge/panels/index.json and the board's gene file,
then checks the prediction the way the published format rules describe:

  * gene names equal to the board's panel, in panel order
  * cell count within the board's min_cells..max_cells (inclusive)
  * X two-dimensional, numeric, finite and non-negative
  * X plausibly log-normalised (warnings only; see --cp10k-tol)
  * obsm['spatial_3D'] present, shape (n_cells, >=3) and finite when the board needs coordinates
  * obs / obsm keys the contract lists as required
  * file extension .h5ad and size under the 1,200 MiB upload limit

Nothing is uploaded and nothing is scored. A PASS here does not guarantee the
server will accept the file; the server's own free format check (vec_submit.py
runs it first) is the authority.

Exit codes: 0 = no failures, 1 = at least one failure (or a warning with --strict),
2 = usage or setup problem (unknown board, index not reachable, file missing).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

import vec_common as vc

PASS, FAIL, WARN, INFO, SKIP = "PASS", "FAIL", "WARN", "INFO", "SKIP"

# log1p(1e4) is about 9.21 and log1p(1e6) about 13.8; values above this are
# very unlikely on a log scale.
LOG_SCALE_MAX = 15.0
CP10K_TARGET = 1e4
ROW_CHUNK = 4096
CONSTANT_SAMPLE = 500


@dataclass
class Check:
    name: str
    status: str
    detail: str


@dataclass
class Report:
    file: str
    board: str
    checks: list[Check]
    n_cells: int | None = None
    n_genes: int | None = None
    sha256: str | None = None

    @property
    def n_fail(self) -> int:
        return sum(c.status == FAIL for c in self.checks)

    @property
    def n_warn(self) -> int:
        return sum(c.status == WARN for c in self.checks)

    def ok(self, strict: bool = False) -> bool:
        return self.n_fail == 0 and (not strict or self.n_warn == 0)

    def to_dict(self, strict: bool = False) -> dict[str, Any]:
        d = asdict(self)
        d["result"] = "PASS" if self.ok(strict) else "FAIL"
        d["n_fail"], d["n_warn"] = self.n_fail, self.n_warn
        return d


# ----------------------------------------------------------------- X statistics


def _x_stats(X: Any) -> dict[str, Any]:
    """Finite / sign / integer / expm1-row-sum statistics, computed in row chunks."""
    n_obs = X.shape[0]
    out = {"n_nan": 0, "n_inf": 0, "n_neg": 0, "max": -np.inf, "min": np.inf, "all_integer": True}
    rowsums = np.zeros(n_obs, dtype=np.float64)
    if sp.issparse(X):
        Xc = sp.csr_matrix(X)
        data = np.asarray(Xc.data, dtype=np.float64)
        out["n_nan"] = int(np.isnan(data).sum())
        out["n_inf"] = int(np.isinf(data).sum())
        out["n_neg"] = int((data < 0).sum())
        fin = data[np.isfinite(data)]
        if fin.size:
            out["max"], out["min"] = float(fin.max()), float(fin.min())
            out["all_integer"] = bool(np.all(fin == np.round(fin)))
        if Xc.nnz < Xc.shape[0] * Xc.shape[1]:  # implicit zeros exist
            out["max"], out["min"] = max(out["max"], 0.0), min(out["min"], 0.0)
        with np.errstate(over="ignore", invalid="ignore"):
            ex = sp.csr_matrix((np.expm1(data), Xc.indices, Xc.indptr), shape=Xc.shape)
            rowsums = np.asarray(ex.sum(axis=1), dtype=np.float64).ravel()
    else:
        for i in range(0, n_obs, ROW_CHUNK):
            block = np.asarray(X[i:i + ROW_CHUNK], dtype=np.float64)
            out["n_nan"] += int(np.isnan(block).sum())
            out["n_inf"] += int(np.isinf(block).sum())
            out["n_neg"] += int((block < 0).sum())
            fin = block[np.isfinite(block)]
            if fin.size:
                out["max"] = max(out["max"], float(fin.max()))
                out["min"] = min(out["min"], float(fin.min()))
                if out["all_integer"]:
                    out["all_integer"] = bool(np.all(fin == np.round(fin)))
            with np.errstate(over="ignore", invalid="ignore"):
                rowsums[i:i + ROW_CHUNK] = np.expm1(block).sum(axis=1)
    out["rowsums"] = rowsums
    return out


def _sample_rows_dense(X: Any, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(X.shape[0], size=min(n, X.shape[0]), replace=False))
    block = X[idx]
    return block.toarray() if sp.issparse(block) else np.asarray(block)


# ------------------------------------------------------------------- the checks


def check_file(
    path: Path,
    board_key: str,
    board: dict[str, Any],
    genes: list[str],
    *,
    gene_note: str = "",
    use_veckit: bool = True,
    cp10k_tol: float = 0.01,
) -> Report:
    """Run every check and return a Report. Never raises for a bad file."""
    checks: list[Check] = []
    rep = Report(file=str(path), board=board_key, checks=checks)
    task = board_key.split(":")[0].upper()
    needs_coords = bool(board.get("needs_coords"))

    # --- the contract itself
    if gene_note.startswith("MISMATCH"):
        checks.append(Check("panel", WARN, f"{board.get('genes_file')}: {gene_note}. Try --refresh."))
    else:
        checks.append(Check("panel", INFO, f"{board.get('genes_file')}: {len(genes)} genes, {gene_note or 'not verified'}"))

    # --- file level
    if not path.is_file():
        checks.append(Check("file", FAIL, "file not found"))
        return rep
    size = path.stat().st_size
    rep.sha256 = vc.sha256_file(path)
    if path.suffix.lower() != ".h5ad":
        checks.append(Check("file", FAIL, f"extension {path.suffix!r}; the uploader requires an AnnData .h5ad"))
    elif size == 0:
        checks.append(Check("file", FAIL, "the file is empty"))
    elif size > vc.MAX_UPLOAD_BYTES:
        checks.append(Check("file", FAIL, f"{vc.human_bytes(size)} exceeds the 1,200 MiB upload limit (HTTP 413)"))
    else:
        checks.append(Check("file", PASS, f".h5ad, {vc.human_bytes(size)} (limit 1,200 MiB)"))

    try:
        import anndata as ad

        a = ad.read_h5ad(path)
    except Exception as e:  # report any reader failure as a format failure
        checks.append(Check("read", FAIL, f"cannot read as AnnData: {type(e).__name__}: {e}"))
        return rep
    rep.n_cells, rep.n_genes = int(a.n_obs), int(a.n_vars)
    X = a.X
    if X is None:
        checks.append(Check("read", FAIL, "AnnData has no X matrix"))
        return rep
    kind = type(X).__name__
    checks.append(Check("read", PASS, f"AnnData {a.n_obs} cells x {a.n_vars} genes, X {kind} {X.dtype}"))

    # --- genes
    pred = [str(g) for g in a.var_names]
    hint_t1 = (task == "T1" and len(pred) < 1000)
    if pred == genes:
        checks.append(Check("genes", PASS, f"{len(genes)} names, identical to the panel and in panel order"))
    else:
        sp_, sg = set(pred), set(genes)
        if len(pred) != len(sp_):
            dups = sorted(g for g, c in Counter(pred).items() if c > 1)[:5]
            checks.append(Check("genes", FAIL, f"duplicate gene names, e.g. {dups}"))
        elif sp_ == sg:
            first = next(i for i, (p, g) in enumerate(zip(pred, genes)) if p != g)
            checks.append(Check(
                "genes", FAIL,
                f"same {len(genes)} genes in a different order (first difference at position {first}: "
                f"{pred[first]!r} vs panel {genes[first]!r}). Reorder with adata = adata[:, panel_genes].copy()"))
        else:
            missing = [g for g in genes if g not in sp_][:5]
            extra = [g for g in pred if g not in sg][:5]
            msg = (f"expected {len(genes)} genes, got {len(pred)}; "
                   f"{len(sg - sp_)} missing (first: {missing}), {len(sp_ - sg)} extra (first: {extra})")
            if hint_t1:
                msg += ". Task 1 is whole-transcriptome, not the 500-gene MERFISH panel"
            checks.append(Check("genes", FAIL, msg))

    # --- cells
    lo, hi = board.get("min_cells"), board.get("max_cells")
    n = int(a.n_obs)
    if lo is not None and hi is not None:
        if lo <= n <= hi:
            checks.append(Check("cells", PASS, f"{n} cells, within the board's {lo}..{hi}"))
        else:
            checks.append(Check("cells", FAIL, f"{n} cells, outside the board's {lo}..{hi}; the server rejects it before scoring"))
    else:
        checks.append(Check("cells", WARN, f"{n} cells; the contract lists no min_cells/max_cells"))
    if not a.obs_names.is_unique:
        checks.append(Check("obs names", WARN, "cell names (obs index) are not unique; harmless for scoring, but call adata.obs_names_make_unique()"))

    # --- X values
    if X.ndim != 2:
        checks.append(Check("X", FAIL, f"X must be 2-D, got {X.ndim}-D"))
        return rep
    if not np.issubdtype(np.dtype(X.dtype), np.number) or np.dtype(X.dtype) == np.bool_:
        checks.append(Check("X", FAIL, f"X dtype {X.dtype} is not numeric"))
        return rep
    st = _x_stats(X)
    if st["n_nan"] or st["n_inf"]:
        checks.append(Check("X finite", FAIL, f"{st['n_nan']} NaN and {st['n_inf']} infinite values; a single one rejects the file"))
    else:
        checks.append(Check("X finite", PASS, "no NaN or infinity"))
    if st["n_neg"]:
        checks.append(Check("X non-negative", FAIL, f"{st['n_neg']} negative values (min {st['min']:.4g}); a centred or z-scored matrix is not accepted"))
    else:
        checks.append(Check("X non-negative", PASS, f"min {st['min']:.4g}, max {st['max']:.4g}"))
    if np.dtype(X.dtype) != np.float32:
        checks.append(Check("X dtype", INFO, f"{X.dtype}; the scorer casts to float32, so this is fine"))

    # --- log-normalisation heuristics (warnings, never failures)
    if st["all_integer"] and st["max"] > 1:
        checks.append(Check("X log scale", WARN, "every value is an integer: this looks like raw counts. Values must already be log-normalised; counts pass the format check and are then scored as if they were on the log scale"))
    elif st["max"] > LOG_SCALE_MAX:
        checks.append(Check("X log scale", WARN, f"max value {st['max']:.4g} is unusually large for log-normalised data; was the log1p step skipped?"))
    rs = st["rowsums"]
    if st["n_nan"] or st["n_inf"]:
        checks.append(Check("X row sums", SKIP, "not computed because X has non-finite values"))
    else:
        med = float(np.median(rs))
        dev = np.abs(rs - CP10K_TARGET) / CP10K_TARGET
        n_off = int((dev > cp10k_tol).sum())
        n_empty = int((rs == 0).sum())
        detail = (f"expm1(X) row sums: median {med:.6g}, min {rs.min():.6g}, max {rs.max():.6g}, "
                  f"largest relative deviation from 1e4 {dev.max():.2e}")
        if n_off == 0:
            checks.append(Check("X row sums", PASS, detail + " (consistent with log1p of counts per 10k)"))
        else:
            checks.append(Check("X row sums", WARN, detail + f"; {n_off} of {len(rs)} rows deviate by more than {cp10k_tol:.0%}. "
                                "The site requires log-normalised values but does not state a target sum; "
                                "compare with how the released training files are scaled"))
        if n_empty:
            checks.append(Check("X empty rows", WARN, f"{n_empty} cells have no expression at all"))
    try:
        sample = _sample_rows_dense(X, CONSTANT_SAMPLE)
        if sample.shape[0] > 1 and np.all(np.ptp(sample, axis=0) == 0):
            checks.append(Check("X spread", WARN, "all sampled cells have identical expression; the distribution metrics score a single repeated cell near nothing"))
    except MemoryError:
        checks.append(Check("X spread", SKIP, "not enough memory to sample rows"))

    # --- obs / obsm requirements
    for k in board.get("obs_required") or []:
        checks.append(Check(f"obs['{k}']", PASS if k in a.obs else FAIL, "present" if k in a.obs else "required by the board contract but missing"))
    required_obsm = list(board.get("obsm_required") or [])
    if needs_coords and "spatial_3D" not in required_obsm:
        required_obsm.append("spatial_3D")
    for k in required_obsm:
        if k == "spatial_3D":
            continue
        checks.append(Check(f"obsm['{k}']", PASS if k in a.obsm else FAIL, "present" if k in a.obsm else "required by the board contract but missing"))

    # --- coordinates
    if "spatial_3D" in required_obsm:
        checks.append(_check_coords(a, n))
    elif "spatial_3D" in a.obsm:
        checks.append(Check("coordinates", INFO, "obsm['spatial_3D'] present but this board does not read coordinates"))

    # --- optional: the official veckit loader
    if use_veckit:
        checks.append(_veckit_check(path, genes, needs_coords))
    return rep


def _check_coords(a: Any, n: int) -> Check:
    if "spatial_3D" not in a.obsm:
        return Check("coordinates", FAIL, "obsm['spatial_3D'] is required for this board and missing; a file with expression but no coordinates is rejected before scoring")
    try:
        C = np.asarray(a.obsm["spatial_3D"])
    except Exception as e:  # e.g. an object column that cannot become an array
        return Check("coordinates", FAIL, f"obsm['spatial_3D'] cannot be read as an array: {e}")
    if C.ndim != 2 or C.shape[0] != n:
        return Check("coordinates", FAIL, f"obsm['spatial_3D'] has shape {C.shape}; expected ({n}, 3) or more columns")
    if C.shape[1] < 3:
        return Check("coordinates", FAIL, f"obsm['spatial_3D'] has {C.shape[1]} columns; at least 3 (x, y, z) are required")
    if not np.issubdtype(C.dtype, np.number):
        return Check("coordinates", FAIL, f"obsm['spatial_3D'] dtype {C.dtype} is not numeric")
    C3 = C[:, :3].astype(np.float64)
    bad = int((~np.isfinite(C3)).sum())
    if bad:
        return Check("coordinates", FAIL, f"obsm['spatial_3D'] has {bad} NaN or infinite values in the first three columns")
    span = np.ptp(C3, axis=0)
    extra = f"; columns beyond the third are ignored" if C.shape[1] > 3 else ""
    if np.all(span == 0):
        return Check("coordinates", WARN, "every cell has the same position" + extra)
    return Check("coordinates", PASS, f"shape {C.shape}, finite, extent x/y/z {span[0]:.4g}/{span[1]:.4g}/{span[2]:.4g}{extra}")


def _veckit_check(path: Path, genes: list[str], needs_coords: bool) -> Check:
    try:
        import score_h5ad  # the official veckit module (pip install veckit)
    except ImportError:
        return Check("veckit", SKIP, "official veckit not importable; install it (pip install veckit) to also run its loader")
    fn = getattr(score_h5ad, "_prediction", None)
    if fn is None:
        return Check("veckit", SKIP, "veckit found, but this version has no score_h5ad._prediction loader to call")
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            ver = version("veckit")
        except PackageNotFoundError:
            ver = f"from {Path(score_h5ad.__file__).parent}"
    except ImportError:
        ver = "unknown version"
    try:
        fn(path, genes, need_coords=needs_coords, allow_reorder=False)
    except Exception as e:  # the loader signals problems with exceptions; report them
        return Check("veckit", FAIL, f"veckit {ver} loader rejected the file: {type(e).__name__}: {e}")
    return Check("veckit", PASS, f"veckit {ver} loader accepted the file")


# ------------------------------------------------------------------------ output


def format_report(rep: Report, board: dict[str, Any], strict: bool = False) -> str:
    lines = [f"vec_check {vc.__version__}  board {rep.board}  ({board.get('label', '')})",
             f"file {rep.file}"]
    if rep.sha256:
        lines.append(f"sha256 {rep.sha256}")
    lo, hi = board.get("min_cells"), board.get("max_cells")
    lines.append(f"contract: {board.get('n_genes')} genes, cells {lo}..{hi}, "
                 f"coordinates {'required' if board.get('needs_coords') else 'not used'}")
    lines.append("")
    w = max(len(c.name) for c in rep.checks) if rep.checks else 10
    for c in rep.checks:
        lines.append(f"  {c.status:<4}  {c.name:<{w}}  {c.detail}")
    lines.append("")
    verdict = "PASS" if rep.ok(strict) else "FAIL"
    lines.append(f"RESULT: {verdict} ({rep.n_fail} failure{'s' * (rep.n_fail != 1)}, "
                 f"{rep.n_warn} warning{'s' * (rep.n_warn != 1)}{', --strict' if strict else ''})")
    if verdict == "PASS":
        lines.append("A local PASS is not a guarantee: run the server's free format check before a scored upload.")
    return "\n".join(lines)


def list_boards(index: dict[str, Any]) -> str:
    rows = [f"{'board':<24} {'genes':>6} {'cells':>13} {'coords':>6}  label"]
    for k, b in index.items():
        rows.append(f"{k:<24} {b.get('n_genes', '?'):>6} {str(b.get('min_cells')) + '..' + str(b.get('max_cells')):>13} "
                    f"{'yes' if b.get('needs_coords') else 'no':>6}  {b.get('label', '')}")
    return "\n".join(rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vec_check.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", nargs="?", type=Path, help="prediction .h5ad")
    p.add_argument("--board", help="board key from the index, e.g. T1:val, T2:heart:val_extrap, T3:gata4")
    p.add_argument("--list-boards", action="store_true", help="print the boards in the index and exit")
    p.add_argument("--index", help="panel index: local index.json path or URL (default: the public index on the site)")
    p.add_argument("--cache-dir", type=Path, help="cache directory (default ~/.cache/vec-kit or $VEC_KIT_CACHE)")
    p.add_argument("--refresh", action="store_true", help="re-download the index and gene file")
    p.add_argument("--offline", action="store_true", help="use only cached files, no network")
    p.add_argument("--no-veckit", action="store_true", help="do not run the official veckit loader even if installed")
    p.add_argument("--cp10k-tol", type=float, default=0.01, help="relative tolerance for the expm1 row-sum check (default 0.01)")
    p.add_argument("--strict", action="store_true", help="treat warnings as failures")
    p.add_argument("--json", action="store_true", help="print the report as JSON")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        src = vc.load_panel_index(args.index, cache_dir=args.cache_dir, refresh=args.refresh, offline=args.offline)
        for note in src.notes:
            print(f"note: {note}", file=sys.stderr)
        if args.list_boards:
            print(list_boards(src.index))
            return 0
        if not args.file or not args.board:
            print("error: give a prediction file and --board (or --list-boards)", file=sys.stderr)
            return 2
        key, board = vc.resolve_board(src.index, args.board)
        genes, note = vc.load_genes(src, board, cache_dir=args.cache_dir, refresh=args.refresh, offline=args.offline)
    except vc.KitError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if not args.file.is_file():
        print(f"error: file not found: {args.file}", file=sys.stderr)
        return 2
    rep = check_file(args.file, key, board, genes, gene_note=note, use_veckit=not args.no_veckit,
                     cp10k_tol=args.cp10k_tol)
    if args.json:
        print(json.dumps(rep.to_dict(args.strict), indent=1))
    else:
        print(format_report(rep, board, args.strict))
    return 0 if rep.ok(args.strict) else 1


if __name__ == "__main__":
    sys.exit(main())
