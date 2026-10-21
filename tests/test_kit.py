"""Tests for the VEC community kit. Standard library unittest; no network needed.

Run:  python -m unittest discover -s tests -v
"""
from __future__ import annotations

import contextlib
import email.parser
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import vec_board  # noqa: E402
import vec_common as vc  # noqa: E402
import vec_submit  # noqa: E402

PY = sys.executable
FAKE_KEY = "test-key-NEVER-PRINT-0123456789"


# --------------------------------------------------------------------- fixtures


def make_index(d: Path) -> Path:
    """A small fake panel index in the same shape as the public one."""
    t1 = [f"Gene{i:03d}" for i in range(40)]  # deliberately not sorted alphabetically below
    t1 = t1[::-1]
    panel = sorted(f"Pan{i:02d}" for i in range(20))
    boards = {
        "T1:val": dict(task="T1", setting=None, split="val", genes=t1, needs_coords=False, obsm_required=[]),
        "T2:heart:val_extrap": dict(task="T2", setting="heart", split="val_extrap", genes=panel,
                                    needs_coords=True, obsm_required=["spatial_3D"]),
        "T3:gata4": dict(task="T3", setting=None, split="val", genes=panel, needs_coords=True,
                         obsm_required=["spatial_3D"], target_ko="gata4"),
        "T2:embryo:val_interp": dict(task="T2", setting="embryo", split="val_interp", genes=panel,
                                     needs_coords=True, obsm_required=["spatial_3D"], bad_hash=True),
    }
    index = {}
    for key, b in boards.items():
        fname = key.replace(":", "__") + ".genes.txt"
        (d / fname).write_text("\n".join(b["genes"]) + "\n")
        digest = hashlib.sha256("\n".join(b["genes"]).encode()).hexdigest()[:16]
        index[key] = {
            "key": key, "label": f"fake {key}", "task": b["task"], "setting": b["setting"], "split": b["split"],
            "target_ko": b.get("target_ko"), "public": True, "n_genes": len(b["genes"]), "genes_file": fname,
            "genes_sha256": "0" * 16 if b.get("bad_hash") else digest, "needs_coords": b["needs_coords"],
            "obs_required": [], "obsm_required": b["obsm_required"], "min_cells": 10, "max_cells": 60,
            "floor_model": "copy_last",
        }
    p = d / "index.json"
    p.write_text(json.dumps(index, indent=1))
    return p


def genes_of(index_path: Path, key: str) -> list[str]:
    b = json.loads(index_path.read_text())[key]
    return (index_path.parent / b["genes_file"]).read_text().split()


def log1p_cp10k(n_cells: int, n_genes: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    counts = rng.poisson(3.0, size=(n_cells, n_genes)).astype(np.float64) + 1
    return np.log1p(counts / counts.sum(1, keepdims=True) * 1e4).astype(np.float32)


def write_h5ad(path: Path, X, genes: list[str], coords: np.ndarray | None = None) -> Path:
    import anndata as ad
    import pandas as pd

    a = ad.AnnData(X=X, obs=pd.DataFrame(index=[f"c{i}" for i in range(X.shape[0])]),
                   var=pd.DataFrame(index=genes))
    if coords is not None:
        a.obsm["spatial_3D"] = coords
    a.write_h5ad(path)
    return path


def run_check(file: Path, board: str, index: Path, *extra: str) -> tuple[int, dict]:
    cmd = [PY, str(ROOT / "vec_check.py"), str(file), "--board", board, "--index", str(index),
           "--no-veckit", "--json", *extra]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    try:
        rep = json.loads(p.stdout) if p.stdout.strip() else {}
    except ValueError:
        raise AssertionError(f"not JSON: {p.stdout}\n{p.stderr}")
    return p.returncode, rep


def statuses(rep: dict) -> dict[str, str]:
    return {c["name"]: c["status"] for c in rep.get("checks", [])}


# ------------------------------------------------------------------ vec_check


class VecCheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="vec-kit-test-"))
        cls.index = make_index(cls.tmp)
        cls.panel = genes_of(cls.index, "T2:heart:val_extrap")
        cls.t1 = genes_of(cls.index, "T1:val")
        rng = np.random.default_rng(1)
        cls.coords = rng.normal(size=(30, 3)).astype(np.float32) * 100

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def f(self, name: str) -> Path:
        return self.tmp / name

    def test_pass_t2_dense(self):
        p = write_h5ad(self.f("ok_t2.h5ad"), log1p_cp10k(30, 20), self.panel, self.coords)
        code, rep = run_check(p, "T2:heart:val_extrap", self.index)
        self.assertEqual(code, 0, rep)
        self.assertEqual(rep["n_fail"], 0)
        self.assertEqual(rep["n_warn"], 0)
        self.assertEqual(statuses(rep)["X row sums"], "PASS")

    def test_pass_t1_sparse_and_board_case(self):
        p = write_h5ad(self.f("ok_t1.h5ad"), sp.csr_matrix(log1p_cp10k(25, 40)), self.t1)
        code, rep = run_check(p, "t1:VAL", self.index)
        self.assertEqual(code, 0, rep)
        self.assertNotIn("coordinates", statuses(rep))

    def test_pass_t3_extra_coordinate_columns(self):
        c4 = np.hstack([self.coords, np.zeros((30, 1), np.float32)])
        p = write_h5ad(self.f("ok_t3.h5ad"), log1p_cp10k(30, 20), self.panel, c4)
        code, rep = run_check(p, "T3:gata4", self.index)
        self.assertEqual(code, 0, rep)
        self.assertIn("ignored", [c for c in rep["checks"] if c["name"] == "coordinates"][0]["detail"])

    def test_fail_gene_order(self):
        p = write_h5ad(self.f("order.h5ad"), log1p_cp10k(30, 20), self.panel[::-1], self.coords)
        code, rep = run_check(p, "T2:heart:val_extrap", self.index)
        self.assertEqual(code, 1)
        self.assertEqual(statuses(rep)["genes"], "FAIL")
        self.assertIn("different order", [c for c in rep["checks"] if c["name"] == "genes"][0]["detail"])

    def test_fail_missing_gene(self):
        g = self.panel[:-1] + ["NotAGene"]
        p = write_h5ad(self.f("missing.h5ad"), log1p_cp10k(30, 20), g, self.coords)
        code, rep = run_check(p, "T2:heart:val_extrap", self.index)
        self.assertEqual(code, 1)
        self.assertEqual(statuses(rep)["genes"], "FAIL")

    def test_fail_t1_given_panel_genes(self):
        p = write_h5ad(self.f("t1_panel.h5ad"), log1p_cp10k(30, 20), self.panel)
        code, rep = run_check(p, "T1:val", self.index)
        self.assertEqual(code, 1)
        self.assertIn("whole-transcriptome", [c for c in rep["checks"] if c["name"] == "genes"][0]["detail"])

    def test_fail_nan_and_inf(self):
        X = log1p_cp10k(30, 20)
        X[0, 0], X[1, 1] = np.nan, np.inf
        p = write_h5ad(self.f("nan.h5ad"), X, self.panel, self.coords)
        code, rep = run_check(p, "T2:heart:val_extrap", self.index)
        self.assertEqual(code, 1)
        self.assertEqual(statuses(rep)["X finite"], "FAIL")
        self.assertEqual(statuses(rep)["X row sums"], "SKIP")

    def test_fail_negative(self):
        X = log1p_cp10k(30, 20)
        X -= X.mean(0, keepdims=True)  # centred, a common mistake
        p = write_h5ad(self.f("neg.h5ad"), X, self.panel, self.coords)
        code, rep = run_check(p, "T2:heart:val_extrap", self.index)
        self.assertEqual(code, 1)
        self.assertEqual(statuses(rep)["X non-negative"], "FAIL")

    def test_fail_cell_count_low_and_high(self):
        lo = write_h5ad(self.f("few.h5ad"), log1p_cp10k(9, 20), self.panel, self.coords[:9])
        hi = write_h5ad(self.f("many.h5ad"), log1p_cp10k(61, 20), self.panel,
                        np.random.default_rng(0).normal(size=(61, 3)))
        for p in (lo, hi):
            code, rep = run_check(p, "T2:heart:val_extrap", self.index)
            self.assertEqual(code, 1)
            self.assertEqual(statuses(rep)["cells"], "FAIL")
        # both limits are inclusive
        edge = write_h5ad(self.f("edge.h5ad"), log1p_cp10k(10, 20), self.panel, self.coords[:10])
        code, rep = run_check(edge, "T2:heart:val_extrap", self.index)
        self.assertEqual(statuses(rep)["cells"], "PASS")

    def test_fail_missing_coordinates(self):
        p = write_h5ad(self.f("nocoord.h5ad"), log1p_cp10k(30, 20), self.panel)
        code, rep = run_check(p, "T2:heart:val_extrap", self.index)
        self.assertEqual(code, 1)
        self.assertEqual(statuses(rep)["coordinates"], "FAIL")

    def test_fail_two_dimensional_coordinates(self):
        p = write_h5ad(self.f("c2.h5ad"), log1p_cp10k(30, 20), self.panel, self.coords[:, :2])
        code, rep = run_check(p, "T3:gata4", self.index)
        self.assertEqual(code, 1)
        self.assertEqual(statuses(rep)["coordinates"], "FAIL")

    def test_fail_nan_coordinates(self):
        c = self.coords.copy()
        c[3, 2] = np.nan
        p = write_h5ad(self.f("cnan.h5ad"), log1p_cp10k(30, 20), self.panel, c)
        code, rep = run_check(p, "T2:heart:val_extrap", self.index)
        self.assertEqual(code, 1)
        self.assertEqual(statuses(rep)["coordinates"], "FAIL")

    def test_warn_raw_counts_and_strict(self):
        counts = np.random.default_rng(2).poisson(5, size=(30, 20)).astype(np.float32) + 1
        p = write_h5ad(self.f("counts.h5ad"), counts, self.panel, self.coords)
        code, rep = run_check(p, "T2:heart:val_extrap", self.index)
        self.assertEqual(code, 0)  # format-valid, so only warnings
        self.assertEqual(statuses(rep)["X log scale"], "WARN")
        self.assertEqual(statuses(rep)["X row sums"], "WARN")
        code, _ = run_check(p, "T2:heart:val_extrap", self.index, "--strict")
        self.assertEqual(code, 1)

    def test_warn_other_target_sum(self):
        rng = np.random.default_rng(3)
        counts = rng.poisson(3.0, size=(30, 20)).astype(np.float64) + 1
        X = np.log1p(counts / counts.sum(1, keepdims=True) * 1e6).astype(np.float32)  # log1p CPM
        p = write_h5ad(self.f("cpm.h5ad"), X, self.panel, self.coords)
        code, rep = run_check(p, "T2:heart:val_extrap", self.index)
        self.assertEqual(code, 0)
        self.assertEqual(statuses(rep)["X row sums"], "WARN")

    def test_warn_identical_cells(self):
        X = np.tile(log1p_cp10k(1, 20), (30, 1))
        p = write_h5ad(self.f("same.h5ad"), X, self.panel, self.coords)
        code, rep = run_check(p, "T2:heart:val_extrap", self.index)
        self.assertEqual(statuses(rep)["X spread"], "WARN")

    def test_warn_gene_hash_mismatch(self):
        p = write_h5ad(self.f("emb.h5ad"), log1p_cp10k(30, 20), self.panel, self.coords)
        code, rep = run_check(p, "T2:embryo:val_interp", self.index)
        self.assertEqual(code, 0)
        self.assertEqual(statuses(rep)["panel"], "WARN")

    def test_fail_wrong_extension(self):
        src = write_h5ad(self.f("ext.h5ad"), log1p_cp10k(30, 20), self.panel, self.coords)
        p = self.f("ext.hdf5")
        shutil.copy(src, p)
        code, rep = run_check(p, "T2:heart:val_extrap", self.index)
        self.assertEqual(code, 1)
        self.assertEqual(statuses(rep)["file"], "FAIL")

    def test_fail_not_h5ad_content(self):
        p = self.f("garbage.h5ad")
        p.write_bytes(b"not an hdf5 file")
        code, rep = run_check(p, "T2:heart:val_extrap", self.index)
        self.assertEqual(code, 1)
        self.assertEqual(statuses(rep)["read"], "FAIL")

    def test_unknown_board_is_usage_error(self):
        p = write_h5ad(self.f("ub.h5ad"), log1p_cp10k(30, 20), self.panel, self.coords)
        code, _ = run_check(p, "T9:nope", self.index)
        self.assertEqual(code, 2)


# ------------------------------------------------------------------ vec_submit


class FakeApi(BaseHTTPRequestHandler):
    """A tiny stand-in for the challenge API, recording every request."""

    state: dict = {}

    def log_message(self, *a):  # keep test output quiet
        pass

    def _send(self, code: int, obj) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authed(self) -> bool:
        if self.headers.get("Authorization") != f"Bearer {FAKE_KEY}":
            self._send(401, {"detail": "bad key"})
            return False
        return True

    def do_GET(self):
        s = self.state
        s["requests"].append(("GET", self.path, dict(self.headers)))
        if self.path == "/challenge/phase":
            return self._send(200, {"phase": "p2", "label": "P2 test", "accepts_submissions": True, "split": "val"})
        if self.path == "/challenge/boards":
            return self._send(200, {"split": "val", "boards": [{"key": "T2:heart:val_extrap", "task": "T2"}]})
        if not self._authed():
            return
        if self.path == "/challenge/integration/quota":
            return self._send(200, {"daily": {"T2": {"used": s["used"], "limit": 8, "uploads": s["uploads"], "upload_limit": 16}}})
        if self.path.startswith("/challenge/submissions/"):
            sid = self.path.rsplit("/", 1)[1]
            sub = s["subs"].get(sid)
            if not sub:
                return self._send(404, {"detail": "not found"})
            sub["polls"] += 1
            status = "scoring" if sub["polls"] < 2 else "scored"
            out = {"id": sid, "task": "T2", "status": status, "bundle_key": "T2:heart:val_extrap",
                   "sha256": sub["sha256"], "headline": 61.5 if status == "scored" else None,
                   "metrics": {"de_score": 55.0, "mmd_u": 70.0} if status == "scored" else None,
                   "rank": 3, "board_size": 10, "reject_reason": None, "shows_scores": True}
            return self._send(200, out)
        self._send(404, {"detail": "no route"})

    def do_POST(self):
        s = self.state
        n = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(n)
        msg = email.parser.BytesParser().parsebytes(
            b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\n\r\n" + raw)
        form, file_bytes, filename = {}, b"", None
        for part in msg.get_payload():
            name = part.get_param("name", header="content-disposition")
            if part.get_filename():
                filename, file_bytes = part.get_filename(), part.get_payload(decode=True)
            else:
                form[name] = part.get_payload(decode=True).decode()
        idem = self.headers.get("Idempotency-Key")
        s["requests"].append(("POST", self.path, dict(self.headers), form, len(file_bytes)))
        if not self._authed():
            return
        if form.get("format_only") == "true":
            ok = file_bytes[:8] == b"\x89HDF\r\n\x1a\n"
            return self._send(200, {"ok": ok, "problems": [] if ok else ["not an HDF5 file"], "format_only": True})
        if s["fail_next_post"] > 0:
            s["fail_next_post"] -= 1
            return self._send(503, {"detail": "try again"})
        sha = hashlib.sha256(file_bytes).hexdigest()
        if idem in s["by_key"]:
            prev = s["by_key"][idem]
            if prev["payload"] != (sha, filename, json.dumps(form, sort_keys=True)):
                return self._send(409, {"detail": "idempotency conflict"})
            return self._send(200, prev["response"])
        s["used"] += 1
        s["uploads"] += 1
        sid = f"sub-{len(s['subs']) + 1}"
        s["subs"][sid] = {"polls": 0, "sha256": sha}
        resp = {"id": sid, "status": "queued", "task": "T2", "bundle_key": "T2:heart:val_extrap", "bytes": len(file_bytes),
                "sha256": sha, "problems": [], "used_credit": False}
        s["by_key"][idem] = {"payload": (sha, filename, json.dumps(form, sort_keys=True)), "response": resp}
        self._send(200, resp)


class VecSubmitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="vec-kit-submit-"))
        cls.index = make_index(cls.tmp)
        panel = genes_of(cls.index, "T2:heart:val_extrap")
        coords = np.random.default_rng(1).normal(size=(30, 3)).astype(np.float32)
        cls.pred = write_h5ad(cls.tmp / "pred.h5ad", log1p_cp10k(30, 20), panel, coords)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        FakeApi.state = {"requests": [], "subs": {}, "by_key": {}, "used": 0, "uploads": 0, "fail_next_post": 0}
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeApi)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.env = {"VE_API_BASE": f"http://127.0.0.1:{self.server.server_address[1]}",
                    vc.API_KEY_ENV: FAKE_KEY}
        self.old_env = {k: os.environ.get(k) for k in self.env}
        os.environ.update(self.env)
        self.old_backoff = vec_submit.BACKOFF_S
        vec_submit.BACKOFF_S = (0.01, 0.01, 0.01)
        self.ledger = self.tmp / f"ledger_{self._testMethodName}.jsonl"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        vec_submit.BACKOFF_S = self.old_backoff
        for k, v in self.old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def submit(self, *extra: str) -> tuple[int, str]:
        out, err = io.StringIO(), io.StringIO()
        argv = [str(self.pred), "--board", "T2:heart:val_extrap", "--model", "unit-test", "--index", str(self.index),
                "--ledger", str(self.ledger), "--poll-interval", "0.05", "--max-wait", "10", *extra]
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = vec_submit.main(argv)
        text = out.getvalue() + err.getvalue()
        self.assertNotIn(FAKE_KEY, text)
        if self.ledger.exists():
            self.assertNotIn(FAKE_KEY, self.ledger.read_text())
        return code, text

    def posts(self):
        return [r for r in FakeApi.state["requests"] if r[0] == "POST"]

    def test_idempotency_key(self):
        a = vec_submit.idempotency_key("T1:val", "m", "ab" * 32)
        self.assertEqual(a, vec_submit.idempotency_key("T1:val", "m", "AB" * 32))
        self.assertNotEqual(a, vec_submit.idempotency_key("T1:val", "m2", "ab" * 32))
        self.assertNotEqual(a, vec_submit.idempotency_key("T3:gata4", "m", "ab" * 32))
        self.assertNotEqual(a, vec_submit.idempotency_key("T1:val", "m", "ab" * 32, tag="2"))
        self.assertTrue(1 <= len(a) <= 128)
        self.assertTrue(set(a) <= set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-"))

    def test_board_fields(self):
        self.assertEqual(vc.board_form_fields("T1:val"), {"task": "T1"})
        self.assertEqual(vc.board_form_fields("T2:heart:val_extrap"), {"task": "T2", "setting": "heart", "mode": "extrap"})
        self.assertEqual(vc.board_form_fields("T2:embryo:val_interp"), {"task": "T2", "setting": "embryo", "mode": "interp"})
        self.assertEqual(vc.board_form_fields("T3:gata4"), {"task": "T3", "target": "gata4"})
        with self.assertRaises(vc.KitError):
            vc.board_form_fields("T2:heart")

    def test_multipart_length_is_exact(self):
        body = vec_submit.MultipartBody([("task", "T2"), ("model", 'a "quoted" name')], "file", self.pred)
        data = b"".join(body)
        self.assertEqual(len(data), body.length)
        self.assertEqual(data, b"".join(body))  # re-iterable for retries

    def test_dry_run_sends_nothing(self):
        code, text = self.submit("--dry-run", "--scored")
        self.assertEqual(code, 0)
        self.assertEqual(FakeApi.state["requests"], [])
        self.assertIn("Idempotency-Key: vkit-", text)
        self.assertIn("redacted", text)

    def test_format_check_only_by_default(self):
        code, text = self.submit()
        self.assertEqual(code, 0)
        posts = self.posts()
        self.assertEqual(len(posts), 1)
        _, _, headers, form, nbytes = posts[0]
        self.assertEqual(form, {"task": "T2", "setting": "heart", "mode": "extrap", "format_only": "true"})
        self.assertNotIn("Idempotency-Key", headers)
        self.assertEqual(nbytes, self.pred.stat().st_size)
        self.assertEqual(FakeApi.state["used"], 0)
        self.assertIn("No scored attempt spent", text)

    def test_scored_flow_retry_and_no_double_spend(self):
        FakeApi.state["fail_next_post"] = 1  # first scored POST gets a 503
        code, text = self.submit("--scored")
        self.assertEqual(code, 0, text)
        self.assertEqual(FakeApi.state["used"], 1)
        scored = [p for p in self.posts() if "format_only" not in p[3]]
        self.assertEqual(len(scored), 2)  # the 503 and the identical retry
        keys = {p[2].get("Idempotency-Key") for p in scored}
        self.assertEqual(len(keys), 1)
        self.assertEqual(scored[0][3], {"task": "T2", "setting": "heart", "mode": "extrap", "model": "unit-test"})
        self.assertIn("headline 61.5", text)
        self.assertIn("server sha256 matches", text)
        events = [json.loads(line)["event"] for line in self.ledger.read_text().splitlines()]
        self.assertEqual(events[:2], ["format_check", "transport_error"])
        self.assertIn("scored_upload", events)
        self.assertIn("final", events)
        # Running the same command again polls the ledger's id instead of uploading again.
        n_before = len(self.posts())
        code, text = self.submit("--scored")
        self.assertEqual(code, 0)
        self.assertEqual(FakeApi.state["used"], 1)
        self.assertEqual(len([p for p in self.posts()[n_before:] if "format_only" not in p[3]]), 0)
        self.assertIn("polling it instead of resending", text)
        # --resend replays the identical request; the server deduplicates it.
        code, text = self.submit("--scored", "--resend")
        self.assertEqual(code, 0)
        self.assertEqual(FakeApi.state["used"], 1)

    def test_format_failure_stops_before_scoring(self):
        bad = self.tmp / "bad.h5ad"
        bad.write_bytes(b"definitely not hdf5")
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            code = vec_submit.main([str(bad), "--board", "T2:heart:val_extrap", "--index", str(self.index),
                                    "--ledger", str(self.ledger), "--scored", "--skip-local-check"])
        self.assertEqual(code, 1)
        self.assertEqual(FakeApi.state["used"], 0)
        self.assertIn("FORMAT CHECK FAILED", out.getvalue())

    def test_local_check_failure_sends_nothing(self):
        bad = write_h5ad(self.tmp / "few.h5ad", log1p_cp10k(5, 20), genes_of(self.index, "T2:heart:val_extrap"),
                         np.zeros((5, 3), np.float32))
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            code = vec_submit.main([str(bad), "--board", "T2:heart:val_extrap", "--index", str(self.index),
                                    "--ledger", str(self.ledger), "--scored"])
        self.assertEqual(code, 1)
        self.assertEqual(FakeApi.state["requests"], [])

    def test_missing_key_is_usage_error(self):
        os.environ.pop(vc.API_KEY_ENV, None)
        code, text = self.submit()
        self.assertEqual(code, 2)
        self.assertEqual(self.posts(), [])

    def test_key_file(self):
        os.environ.pop(vc.API_KEY_ENV, None)
        kf = self.tmp / "key.env"
        kf.write_text(f"# my key\nexport VE_CHALLENGE_API_KEY={FAKE_KEY}\n")
        kf.chmod(0o600)
        code, text = self.submit("--key-file", str(kf))
        self.assertEqual(code, 0, text)


# ------------------------------------------------------------------- vec_board


class VecBoardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="vec-kit-board-"))
        boards = [{"key": "T1:val", "task": "T1"}, {"key": "T2:a:val_interp", "task": "T2"},
                  {"key": "T2:b:val_extrap", "task": "T2"}, {"key": "T3:x", "task": "T3"}]

        def row(team, track, rank, b):
            tasks = {"T1": b.get("T1:val", 0), "T2": (b.get("T2:a:val_interp", 0) + b.get("T2:b:val_extrap", 0)) / 2,
                     "T3": b.get("T3:x", 0)}
            return {"team": team, "track": track, "rank": rank, "boards": b, "tasks": tasks, "total": sum(tasks.values())}

        rows = [row("Team A", "human", 1, {"T1:val": 70, "T2:a:val_interp": 60, "T2:b:val_extrap": 80, "T3:x": 75}),
                row("Team B", "agent", 1, {"T1:val": 65, "T3:x": 70}),
                row("Team C", "human", 2, {"T2:a:val_interp": 90})]
        self.overall = {"phase": "p2", "split": "val", "boards": boards, "rows": rows, "reference": [], "partners": []}
        (self.tmp / "overall.json").write_text(json.dumps(self.overall))
        (self.tmp / "phase.json").write_text(json.dumps({"label": "P2 test"}))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_board(self, *args: str) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            code = vec_board.main(["--from", str(self.tmp), *args])
        return code, out.getvalue()

    def test_totals_verified(self):
        n, bad = vec_board.verify_totals(self.overall)
        self.assertEqual((n, bad), (3, []))
        code, text = self.run_board("overall")
        self.assertEqual(code, 0)
        self.assertIn("all 3 rows", text)
        self.assertIn("Agent Team track: 1 teams", text)

    def test_missing_board_counts_as_zero(self):
        tasks, total = vec_board.recompute(self.overall["rows"][2], vec_board.task_of(self.overall["boards"]))
        self.assertEqual(tasks, {"T1": 0.0, "T2": 45.0, "T3": 0.0})
        self.assertEqual(total, 45.0)

    def test_tampered_total_is_reported(self):
        self.overall["rows"][0]["total"] += 1.0
        (self.tmp / "overall.json").write_text(json.dumps(self.overall))
        code, text = self.run_board("overall")
        self.assertIn("TOTAL CHECK: 1 of 3 rows disagree", text)

    def test_missing_snapshot_file_is_error(self):
        code, text = self.run_board("task", "T1")
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
