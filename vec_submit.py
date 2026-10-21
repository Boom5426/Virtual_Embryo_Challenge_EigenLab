#!/usr/bin/env python3
"""vec_submit: safe uploader for VEC predictions.

What it does, in order:

  1. runs vec_check locally (skip with --skip-local-check);
  2. prints your quota (GET /challenge/integration/quota);
  3. runs the server's free format check (POST /challenge/submissions with
     format_only=true and no Idempotency-Key; no submission is created and no
     attempt is spent);
  4. only with --scored: uploads for scoring with an Idempotency-Key derived from
     board + model + file SHA-256, so an identical retry never spends a second
     attempt; retries timeouts and 5xx with 2, 4, 8 s back-off and never retries
     4xx responses;
  5. polls GET /challenge/submissions/{id} until scored or rejected, compares the
     server's SHA-256 with the local one, and prints the quota again.

Every request and response (never the key) is appended to a local JSONL ledger.

The key is read from --key-file or the VE_CHALLENGE_API_KEY environment variable.
It is sent only in the Authorization header, never printed, never written to the
ledger and never placed on a command line. --dry-run prints the requests and
sends nothing at all.

Exit codes: 0 = format check passed (or scored), 1 = rejected / failed, 2 = usage or
setup problem, 3 = waiting on something outside this tool (evidence, review, timeout).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import sys
import time
import urllib.error
from pathlib import Path
from typing import Any, Iterator

import vec_common as vc

FINAL_STATES = {"scored", "rejected", "cancelled"}
HANDOFF_STATES = {"awaiting_evidence", "eligibility_review"}
BACKOFF_S = (2, 4, 8)
DEFAULT_LEDGER = "vec_ledger.jsonl"
_IDEM_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-")


# ------------------------------------------------------------------ idempotency


def idempotency_key(board: str, model: str, file_sha256: str, tag: str = "") -> str:
    """Deterministic key: the same board, model name and file bytes give the same key.

    The API accepts 1-128 letters, digits or `._:-`. The result is
    'vkit-' + 48 hex characters, well inside that.
    """
    payload = "\n".join([board, model, file_sha256.lower(), tag])
    key = "vkit-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:48]
    assert 1 <= len(key) <= 128 and set(key) <= _IDEM_CHARS
    return key


# -------------------------------------------------------------------- multipart


class MultipartBody:
    """A multipart/form-data body streamed from disk, with a known length.

    Iterating it again restarts from the beginning, so the same object can be
    resent on a retry with identical bytes.
    """

    def __init__(self, fields: list[tuple[str, str]], file_field: str, path: Path, boundary: str | None = None):
        self.path = path
        self.boundary = boundary or "vkit" + secrets.token_hex(16)
        b = self.boundary
        parts = []
        for name, value in fields:
            parts.append(f"--{b}\r\nContent-Disposition: form-data; name=\"{_q(name)}\"\r\n\r\n{value}\r\n")
        parts.append(f"--{b}\r\nContent-Disposition: form-data; name=\"{_q(file_field)}\"; "
                     f"filename=\"{_q(path.name)}\"\r\nContent-Type: application/octet-stream\r\n\r\n")
        self.head = "".join(parts).encode("utf-8")
        self.tail = f"\r\n--{b}--\r\n".encode("utf-8")
        self.length = len(self.head) + path.stat().st_size + len(self.tail)

    @property
    def content_type(self) -> str:
        return f"multipart/form-data; boundary={self.boundary}"

    def __iter__(self) -> Iterator[bytes]:
        yield self.head
        with open(self.path, "rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                yield chunk
        yield self.tail

    def __len__(self) -> int:
        return self.length


def _q(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\r", " ").replace("\n", " ")


# ------------------------------------------------------------------------ client


class Client:
    """Authenticated calls. The key lives only in this object's headers."""

    def __init__(self, base: str, key: str, timeout: float = 600.0):
        self.base = base
        self._auth = {"Authorization": f"Bearer {key}"}
        self.timeout = timeout

    def get(self, path: str) -> vc.HttpResponse:
        return vc.http_request("GET", self.base + path, headers=self._auth, timeout=60)

    def post_submission(self, body: MultipartBody, idem: str | None) -> vc.HttpResponse:
        headers = dict(self._auth)
        headers["Content-Type"] = body.content_type
        headers["Content-Length"] = str(body.length)
        if idem:
            headers["Idempotency-Key"] = idem
        return vc.http_request("POST", self.base + "/challenge/submissions", headers=headers,
                               data=body, timeout=self.timeout)


def post_with_retries(client: Client, body: MultipartBody, idem: str | None, log: Ledger, ctx: dict[str, Any]) -> vc.HttpResponse:
    """Send; on timeout, network error or 5xx, back off 2/4/8 s and resend identically."""
    attempt = 0
    while True:
        try:
            resp = client.post_submission(body, idem)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            err = f"{type(e).__name__}: {e}"
            resp = None
        else:
            err = None
            if resp.status < 500:
                return resp
            err = f"HTTP {resp.status}: {resp.detail()}"
        log.write({**ctx, "event": "transport_error", "attempt": attempt + 1, "error": err})
        if attempt >= len(BACKOFF_S):
            if resp is not None:
                return resp
            raise vc.KitError(f"upload failed after {attempt + 1} tries: {err}. "
                              "Rerunning the same command is safe: the Idempotency-Key is unchanged.")
        wait = BACKOFF_S[attempt]
        print(f"  transient failure ({err}); retrying the identical request in {wait} s", file=sys.stderr)
        time.sleep(wait)
        attempt += 1


# ------------------------------------------------------------------------ ledger


class Ledger:
    """Append-only JSONL record of what was sent and what came back."""

    def __init__(self, path: Path | None):
        self.path = path

    def write(self, rec: dict[str, Any]) -> None:
        if self.path is None:
            return
        rec = {"ts": vc.utc_now(), **rec}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")

    def find_upload(self, idem: str, api: str) -> dict[str, Any] | None:
        """Last scored upload with this key on this API that returned a submission id."""
        if self.path is None or not self.path.exists():
            return None
        hit = None
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if (rec.get("event") == "scored_upload" and rec.get("idempotency_key") == idem
                        and rec.get("api") == api and (rec.get("response") or {}).get("id")):
                    hit = rec
        return hit


# ----------------------------------------------------------------------- display


def format_quota(q: Any) -> list[str]:
    if not isinstance(q, dict):
        return [f"  {json.dumps(q)[:1000]}"]
    lines = []
    daily = q.get("daily")
    if isinstance(daily, dict):
        for name in sorted(daily):
            v = daily[name]
            if isinstance(v, dict):
                bits = []
                if "used" in v or "limit" in v:
                    bits.append(f"scored {v.get('used')}/{v.get('limit')}")
                if "uploads" in v or "upload_limit" in v:
                    bits.append(f"uploads {v.get('uploads')}/{v.get('upload_limit')}")
                rest = {k: x for k, x in v.items() if k not in ("used", "limit", "uploads", "upload_limit")}
                if rest:
                    bits.append(json.dumps(rest))
                lines.append(f"  {name}: {', '.join(bits) if bits else json.dumps(v)}")
            else:
                lines.append(f"  {name}: {json.dumps(v)}")
    for k in sorted(q):
        if k != "daily":
            lines.append(f"  {k}: {json.dumps(q[k])[:300]}")
    return lines or ["  (empty)"]


def print_scores(d: dict[str, Any]) -> None:
    print(f"  status {d.get('status')}  board {d.get('bundle_key')}  id {d.get('id')}")
    if d.get("reject_reason"):
        print(f"  reject_reason: {d['reject_reason']}")
    for p in d.get("problems") or []:
        print(f"  problem: {p}")
    if d.get("status") == "scored":
        if d.get("shows_scores"):
            print(f"  headline {d.get('headline')}  rank {d.get('rank')} of {d.get('board_size')} (team's best on this board)")
            for k, v in (d.get("metrics") or {}).items():
                print(f"    {vc.METRIC_ABBR.get(k, k):>4} {k:<18} {v}")
        else:
            print("  scored, but scores are not shown in the current phase")


def describe_request(title: str, url: str, key_note: str, body: MultipartBody, fields: list[tuple[str, str]],
                     idem: str | None, sha: str) -> str:
    out = [title, f"  POST {url}", f"  Authorization: Bearer <redacted; {key_note}>",
           f"  Content-Type: multipart/form-data; boundary={body.boundary}"]
    if idem:
        out.append(f"  Idempotency-Key: {idem}")
    for k, v in fields:
        out.append(f"  form {k}={v}")
    out.append(f"  form file=@{body.path.name}  ({body.path.stat().st_size:,} bytes, sha256 {sha})")
    out.append(f"  total body {body.length:,} bytes")
    return "\n".join(out)


# -------------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vec_submit.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", type=Path, help="prediction .h5ad")
    p.add_argument("--board", required=True, help="board key, e.g. T1:val, T2:heart:val_extrap, T3:gata4")
    p.add_argument("--model", default="", help="name for the prediction method, shown with your team")
    p.add_argument("--architecture", help="your team's architecture name (optional)")
    p.add_argument("--agent-framework", help="Agent Teams: framework that ran the agent")
    p.add_argument("--agent-model", help="Agent Teams: the actual model identifier used by the agent")
    p.add_argument("--scored", action="store_true", help="after the free format check, spend one scored attempt")
    p.add_argument("--dry-run", action="store_true", help="print the requests and send nothing")
    p.add_argument("--key-file", help=f"file holding the API key (default: ${vc.API_KEY_ENV})")
    p.add_argument("--ledger", default=DEFAULT_LEDGER, help=f"JSONL ledger path (default ./{DEFAULT_LEDGER}); 'none' disables it")
    p.add_argument("--poll-interval", type=float, default=20.0, help="seconds between status polls (the API guide asks for 15-30)")
    p.add_argument("--max-wait", type=float, default=3600.0, help="stop polling after this many seconds (default 3600)")
    p.add_argument("--attempt-tag", default="", help="change this to make a deliberate new attempt with identical bytes")
    p.add_argument("--resend", action="store_true", help="replay the upload even if the ledger already has its id")
    p.add_argument("--skip-local-check", action="store_true", help="do not run vec_check first")
    p.add_argument("--index", help="panel index path or URL for the local check (default: public index)")
    return p


def _fields(args: argparse.Namespace, board_key: str, scored: bool) -> list[tuple[str, str]]:
    f = list(vc.board_form_fields(board_key).items())
    if scored:
        for name, val in (("model", args.model), ("architecture", args.architecture),
                          ("agent_framework", args.agent_framework), ("agent_model", args.agent_model)):
            if val:
                f.append((name, val))
    else:
        f.append(("format_only", "true"))
    return f


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path: Path = args.file
    ledger = Ledger(None if args.ledger.lower() == "none" else Path(args.ledger))
    api = vc.api_base()
    try:
        if not path.is_file():
            raise vc.KitError(f"file not found: {path}")
        if path.stat().st_size > vc.MAX_UPLOAD_BYTES:
            raise vc.KitError("file exceeds the 1,200 MiB upload limit")
        src = vc.load_panel_index(args.index)
        board_key, contract = vc.resolve_board(src.index, args.board)
        vc.board_form_fields(board_key)  # validates the key shape
    except vc.KitError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.poll_interval < 15:
        print(f"note: polling every {args.poll_interval:g} s; the API guide asks for 15-30 s", file=sys.stderr)

    print(f"file   {path}  ({path.stat().st_size:,} bytes)")
    sha = vc.sha256_file(path)
    idem = idempotency_key(board_key, args.model, sha, args.attempt_tag)
    print(f"sha256 {sha}")
    print(f"board  {board_key}  -> form fields {vc.board_form_fields(board_key)}")

    # 1. local check
    if not args.skip_local_check:
        import vec_check

        genes, note = vc.load_genes(src, contract)
        rep = vec_check.check_file(path, board_key, contract, genes, gene_note=note)
        print("\n" + vec_check.format_report(rep, contract) + "\n")
        if not rep.ok():
            print("Local check failed; nothing was sent. Fix the file or use --skip-local-check.", file=sys.stderr)
            return 1

    fmt_fields = _fields(args, board_key, scored=False)
    up_fields = _fields(args, board_key, scored=True)
    fmt_body = MultipartBody(fmt_fields, "file", path)
    up_body = MultipartBody(up_fields, "file", path)
    url = api + "/challenge/submissions"

    if args.dry_run:
        note = vc.key_available(args.key_file)
        print("DRY RUN: nothing is sent.\n")
        print(f"GET {api}/challenge/integration/quota  (Authorization redacted)\n")
        print(describe_request("Request 1: format check (free, no Idempotency-Key)", url, note, fmt_body, fmt_fields, None, sha))
        if args.scored:
            print()
            print(describe_request("Request 2: scored upload (spends one attempt)", url, note, up_body, up_fields, idem, sha))
            print(f"\nThen GET {api}/challenge/submissions/{{id}} every {args.poll_interval:g} s until scored or rejected,"
                  " and GET /challenge/integration/quota again.")
        else:
            print("\n(no scored upload: add --scored to include it)")
        return 0

    try:
        key, key_src = vc.read_api_key(args.key_file)
    except vc.KitError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"key    read from {key_src}")
    client = Client(api, key)
    ctx = {"api": api, "board": board_key, "file": str(path), "bytes": path.stat().st_size, "sha256": sha,
           "model": args.model, "idempotency_key": idem}

    # 2. phase and quota before
    try:
        phase = vc.get_public_json("/challenge/phase")
        boards = vc.get_public_json("/challenge/boards")
    except (vc.KitError, urllib.error.URLError, OSError) as e:
        print(f"error: cannot read the public phase/boards: {e}", file=sys.stderr)
        return 2
    open_keys = [b.get("key") for b in (boards or {}).get("boards", [])]
    print(f"phase  {phase.get('label')}  accepts_submissions={phase.get('accepts_submissions')}  split={phase.get('split')}")
    if not phase.get("accepts_submissions"):
        print("error: the current phase does not accept submissions", file=sys.stderr)
        return 2
    if board_key not in open_keys:
        print(f"error: {board_key} is not an open board now; open boards: {open_keys}", file=sys.stderr)
        return 2
    quota_before = _quota(client, "quota before")

    # 3. free format check
    print("\nServer format check (format_only=true; spends nothing) ...")
    try:
        resp = post_with_retries(client, fmt_body, None, ledger, {**ctx, "stage": "format_check"})
    except vc.KitError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    data = _json_or_none(resp)
    ledger.write({**ctx, "event": "format_check", "fields": dict(fmt_fields), "http": resp.status, "response": data})
    if resp.status != 200 or not isinstance(data, dict):
        print(f"  HTTP {resp.status}: {resp.detail()}")
        return 2 if resp.status in (401, 403) else 1
    for p in data.get("problems") or []:
        print(f"  problem: {p}")
    if not data.get("ok"):
        print("  FORMAT CHECK FAILED. Nothing was created and no attempt was spent.")
        return 1
    print("  format check passed (this does not guarantee scorer acceptance)")
    if not args.scored:
        print("\nNo scored attempt spent. Rerun with --scored to submit this file for scoring.")
        return 0

    # 4. scored upload, at most once per (board, model, bytes)
    prev = ledger.find_upload(idem, api)
    if prev and not args.resend:
        sub_id = prev["response"]["id"]
        print(f"\nThe ledger already holds this upload (id {sub_id}, {prev['ts']}); polling it instead of resending."
              "\n(--resend replays it; the server then returns the original response and spends nothing.)")
    else:
        print(f"\nScored upload with Idempotency-Key {idem} ...")
        try:
            resp = post_with_retries(client, up_body, idem, ledger, {**ctx, "stage": "scored_upload"})
        except vc.KitError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        data = _json_or_none(resp)
        ledger.write({**ctx, "event": "scored_upload", "fields": dict(up_fields), "http": resp.status, "response": data})
        if not 200 <= resp.status < 300 or not isinstance(data, dict):
            print(f"  HTTP {resp.status}: {resp.detail()}")
            if resp.status == 409:
                print("  409: Rules/Terms need acceptance, submissions are closed, or this key was used with a different payload.")
            if resp.status == 429:
                print("  429: a daily scoring/upload limit or final-phase board budget is exhausted.")
            return 1
        sub_id = data.get("id")
        print(f"  accepted as id {sub_id}, status {data.get('status')}, used_credit {data.get('used_credit')}")
        for p in data.get("problems") or []:
            print(f"  problem: {p}")
        if not sub_id:
            print("  the response has no submission id; see the ledger for the full response")
            return 1

    # 5. poll
    final, why = _poll(client, sub_id, args.poll_interval, args.max_wait)
    ledger.write({**ctx, "event": "final", "id": sub_id, "response": final, "note": why})
    print("\nResult:")
    if final is None:
        print(f"  {why}")
        _quota(client, "quota after")
        return 1 if why.startswith("stopped") else 3
    print_scores(final)
    srv_sha = final.get("sha256")
    if srv_sha:
        print(f"  server sha256 {'matches' if srv_sha == sha else 'DIFFERS FROM'} the local file")
    quota_after = _quota(client, "quota after")
    ledger.write({**ctx, "event": "quota", "before": quota_before, "after": quota_after})
    status = final.get("status")
    if status == "scored":
        return 0
    if status in HANDOFF_STATES:
        if status == "awaiting_evidence":
            print("  Agent Team entry: attach evidence to this id (portal or POST /challenge/teams/evidence/{id}).")
        return 3
    return 1


def _json_or_none(resp: vc.HttpResponse) -> Any:
    try:
        return resp.json()
    except (ValueError, UnicodeDecodeError):
        return None


def _quota(client: Client, title: str) -> Any:
    try:
        r = client.get("/challenge/integration/quota")
    except (urllib.error.URLError, OSError) as e:
        print(f"{title}: unavailable ({e})")
        return None
    if r.status != 200:
        print(f"{title}: HTTP {r.status}: {r.detail()}")
        return None
    q = _json_or_none(r)
    print(f"{title}:")
    print("\n".join(format_quota(q)))
    return q


def _poll(client: Client, sub_id: str, interval: float, max_wait: float) -> tuple[dict[str, Any] | None, str]:
    """Return (final status record, '') or (None, reason)."""
    t0 = time.monotonic()
    last = None
    delay = interval
    while time.monotonic() - t0 <= max_wait:
        try:
            r = client.get(f"/challenge/submissions/{sub_id}")
            d = _json_or_none(r) if r.status == 200 else None
            err = None if r.status == 200 else f"HTTP {r.status}: {r.detail()}"
        except (urllib.error.URLError, OSError) as e:
            d, err = None, str(e)
        if isinstance(d, dict):
            delay = interval
            st = d.get("status")
            if st != last:
                print(f"  [{vc.utc_now()}] status {st}")
                last = st
            if st in FINAL_STATES or st in HANDOFF_STATES:
                return d, ""
        else:
            delay = min(delay * 2, 300)  # slower polling during outages
            print(f"  poll failed ({err}); next try in {delay:g} s", file=sys.stderr)
            if err and err.startswith(("HTTP 401", "HTTP 403", "HTTP 404")):
                return None, f"stopped polling submission {sub_id}: {err}"
        time.sleep(delay)
    return None, f"no final status after {max_wait:g} s; rerun the same command later to keep polling"


if __name__ == "__main__":
    sys.exit(main())
