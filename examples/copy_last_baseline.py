#!/usr/bin/env python3
"""Build a floor-model prediction file for one VEC board.

This is the "do nothing" idea the challenge itself uses as its floor:

  * copy_last (T1, T2): predict the preceding observed stage, every cell unchanged,
    carrying its 3D coordinates across on Task 2;
  * wt_identity (T3): predict the matched wild type, cells and coordinates unchanged.

The script takes one released training file and reshapes it into the board's
public contract (panel genes in panel order, a cell count inside the board's
limits, obsm['spatial_3D'] when coordinates are needed). It uses nothing else:
no model, no held-out data.

The only change to the copied cells: if the input has more cells than the board
allows, a uniform random subset is kept (seeded, so the output is reproducible).
That means the file is not guaranteed to score exactly what the published floor
scores. It is a correct, checkable starting point, not a competitive entry.

Example (Task 2 heart extrapolation, validation phase):
    python examples/copy_last_baseline.py --board T2:heart:val_extrap \\
        --input E9.5.h5ad --out copy_last_T2_heart_extrap.h5ad
    python vec_check.py copy_last_T2_heart_extrap.h5ad --board T2:heart:val_extrap
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vec_common as vc  # noqa: E402

# Which released stage to copy for each validation board, from the public Tasks
# page (each target's preceding stage; for T3 the matched wild type at E8.75).
# Check the site for the exact file names in the data manifest.
SUGGESTED_INPUT = {
    "T1:val": "E9.5 single-cell file (target E10.5)",
    "T2:heart:val_interp": "E8.25 heart MERFISH file (target E8.5)",
    "T2:heart:val_extrap": "E9.5 heart MERFISH file (target E10.5)",
    "T2:embryo:val_interp": "E7.25 embryo MERFISH file (target E7.5)",
    "T3:gata4": "E8.75 wild-type MERFISH file (the Gata4 knockout is at E8.75)",
}


def build(input_path: Path, board_key: str, board: dict, genes: list[str], n_cells: int | None, seed: int):
    import anndata as ad
    import pandas as pd

    a = ad.read_h5ad(input_path)
    names = [str(g) for g in a.var_names]
    missing = [g for g in genes if g not in set(names)]
    if missing:
        raise vc.KitError(f"{len(missing)} panel genes are not in the input (first: {missing[:5]}); "
                          "is this the right file for this board?")
    col = {g: i for i, g in enumerate(names)}
    gidx = np.array([col[g] for g in genes])

    lo, hi = int(board["min_cells"]), int(board["max_cells"])
    n = a.n_obs
    target = n_cells if n_cells is not None else min(n, hi)
    if not lo <= target <= hi:
        raise vc.KitError(f"requested {target} cells, but the board allows {lo}..{hi}")
    if target > n:
        raise vc.KitError(f"the input has only {n} cells; cannot copy {target} without inventing cells")
    if target < n:
        rng = np.random.default_rng(seed)
        cidx = np.sort(rng.choice(n, size=target, replace=False))
        print(f"input has {n} cells; keeping a seeded random subset of {target} (board allows {lo}..{hi})")
    else:
        cidx = np.arange(n)

    X = a.X[cidx][:, gidx]
    X = sp.csr_matrix(X, dtype=np.float32) if sp.issparse(X) else np.asarray(X, dtype=np.float32)
    obs = pd.DataFrame(index=pd.Index([str(i) for i in a.obs_names[cidx]], name=None))
    out = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=pd.Index(genes)))
    out.obs_names_make_unique()
    if board.get("needs_coords"):
        if "spatial_3D" not in a.obsm:
            raise vc.KitError("this board needs coordinates but the input has no obsm['spatial_3D']")
        C = np.asarray(a.obsm["spatial_3D"])[cidx, :3].astype(np.float32)
        out.obsm["spatial_3D"] = C
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--board", required=True, help="board key, e.g. T2:heart:val_extrap")
    p.add_argument("--input", type=Path, help="the released stage file to copy (see --hint)")
    p.add_argument("--out", type=Path, help="output .h5ad")
    p.add_argument("--n-cells", type=int, help="number of cells to keep (default: all, capped at the board maximum)")
    p.add_argument("--seed", type=int, default=0, help="seed for the subset (default 0)")
    p.add_argument("--index", help="panel index path or URL (default: the public index)")
    p.add_argument("--hint", action="store_true", help="print which released stage to copy for this board and exit")
    args = p.parse_args(argv)
    try:
        src = vc.load_panel_index(args.index)
        key, board = vc.resolve_board(src.index, args.board)
        floor = board.get("floor_model", "?")
        suggestion = SUGGESTED_INPUT.get(key, "see the Tasks page for this board's preceding stage")
        print(f"board {key}: floor model {floor}; copy the {suggestion}")
        if args.hint:
            return 0
        if not args.input or not args.out:
            raise vc.KitError("give --input and --out (or --hint)")
        if args.out.suffix != ".h5ad":
            raise vc.KitError("--out must end in .h5ad")
        genes, note = vc.load_genes(src, board)
        print(f"panel {board.get('genes_file')}: {len(genes)} genes ({note})")
        out = build(args.input, key, board, genes, args.n_cells, args.seed)
    except vc.KitError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(args.out, compression="gzip")
    print(f"wrote {args.out}: {out.n_obs} cells x {out.n_vars} genes"
          f"{', with spatial_3D' if 'spatial_3D' in out.obsm else ''}")
    print(f"next: python vec_check.py {args.out} --board {key}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
