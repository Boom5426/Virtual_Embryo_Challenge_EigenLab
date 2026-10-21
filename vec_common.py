"""Shared helpers for the VEC community kit.

Public endpoints, the on-disk cache, board keys, API key handling and small
HTTP utilities. Standard library only, so every tool in the kit can import it
without extra dependencies.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__version__ = "0.1.0"

DEFAULT_API_BASE = "https://api.virtualembryo.ai"
DEFAULT_SITE_BASE = "https://virtualembryo.ai"
PANELS_INDEX_PATH = "/challenge/panels/index.json"
API_KEY_ENV = "VE_CHALLENGE_API_KEY"
USER_AGENT = f"vec-community-kit/{__version__} (python-urllib)"

# The official upload limit stated in the API guide: 1,200 MiB per file,
# for scored uploads and format checks alike.
MAX_UPLOAD_BYTES = 1200 * 1024 * 1024

# Seconds after which a cached panel index is re-downloaded.
INDEX_MAX_AGE_S = 24 * 3600

# Metric codes used by the API, with the abbreviations the site uses.
METRIC_ABBR = {
    "de_score": "DES",
    "de_direction": "DCS",
    "severity_slope": "PSS",
    "mmd_u": "MMD",
    "variogram": "CSS",
    "d2_shape": "SDD",
    "occupancy_dice": "ODS",
    "scale_log_ratio": "TSR",
    "neighborhood_mmd": "NFS",
}


class KitError(RuntimeError):
    """A setup or usage problem that should stop the tool with a clear message."""


def api_base() -> str:
    """API root, overridable with VE_API_BASE (the variable the official guide uses)."""
    return os.environ.get("VE_API_BASE", DEFAULT_API_BASE).rstrip("/")


def site_base() -> str:
    """Website root, overridable with VE_SITE_BASE."""
    return os.environ.get("VE_SITE_BASE", DEFAULT_SITE_BASE).rstrip("/")


def default_cache_dir() -> Path:
    """Cache directory: $VEC_KIT_CACHE if set, else ~/.cache/vec-kit."""
    env = os.environ.get("VEC_KIT_CACHE")
    return Path(env).expanduser() if env else Path.home() / ".cache" / "vec-kit"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def human_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{n} B"


# --------------------------------------------------------------------------- HTTP


@dataclass
class HttpResponse:
    status: int
    headers: dict[str, str]
    body: bytes

    def json(self) -> Any:
        if not self.body:
            return None
        return json.loads(self.body.decode("utf-8"))

    def detail(self) -> str:
        """Best-effort error text: the JSON `detail` field the API documents, else raw text."""
        try:
            data = self.json()
        except (ValueError, UnicodeDecodeError):
            return self.body[:500].decode("utf-8", "replace")
        if isinstance(data, dict) and "detail" in data:
            d = data["detail"]
            return d if isinstance(d, str) else json.dumps(d)
        return json.dumps(data)[:500]


def http_request(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    data: Any = None,
    timeout: float = 60.0,
) -> HttpResponse:
    """Send one request. HTTP error statuses are returned, not raised.

    Network failures (DNS, refused connection, timeouts) raise urllib.error.URLError
    or OSError, so callers can decide whether a retry is safe.
    """
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("User-Agent", USER_AGENT)
    req.add_header("Accept", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return HttpResponse(resp.status, dict(resp.headers.items()), resp.read())
    except urllib.error.HTTPError as e:
        body = e.read() if e.fp is not None else b""
        hdrs = dict(e.headers.items()) if e.headers is not None else {}
        return HttpResponse(e.code, hdrs, body)


def get_public_json(path_or_url: str, params: dict[str, str] | None = None, timeout: float = 60.0) -> Any:
    """GET an unauthenticated JSON endpoint. Raises KitError on a non-2xx status."""
    url = path_or_url if path_or_url.startswith("http") else api_base() + path_or_url
    if params:
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    resp = http_request("GET", url, timeout=timeout)
    if not 200 <= resp.status < 300:
        raise KitError(f"GET {url} returned HTTP {resp.status}: {resp.detail()}")
    return resp.json()


# ------------------------------------------------------------------- panel index


def genes_digest(genes: list[str]) -> str:
    """SHA-256 over the newline-joined gene names.

    The public index carries `genes_sha256` as the first 16 hex characters of this
    digest (checked against every board published so far). If the site changes
    that convention, the checker reports a warning rather than failing.
    """
    return hashlib.sha256("\n".join(genes).encode("utf-8")).hexdigest()


@dataclass
class PanelSource:
    """Where the panel index came from, and how to find gene files next to it."""

    index: dict[str, Any]
    location: str  # URL of index.json, or a local directory
    is_local: bool
    notes: list[str] = field(default_factory=list)  # warnings about cache use


def _fresh(path: Path, max_age_s: float) -> bool:
    if not path.exists():
        return False
    age = datetime.now(timezone.utc).timestamp() - path.stat().st_mtime
    return age < max_age_s


def _download(url: str, dest: Path, timeout: float = 60.0) -> None:
    resp = http_request("GET", url, timeout=timeout)
    if resp.status != 200:
        raise KitError(f"GET {url} returned HTTP {resp.status}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(resp.body)
    tmp.replace(dest)


def _cache_name(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:12]


def load_panel_index(
    index: str | None = None,
    *,
    cache_dir: Path | None = None,
    refresh: bool = False,
    offline: bool = False,
) -> PanelSource:
    """Load the board contracts.

    `index` may be a local index.json path (gene files are read from the same
    directory), a URL, or None for the public index on the site.
    """
    if index and not index.startswith(("http://", "https://")):
        p = Path(index).expanduser()
        if not p.is_file():
            raise KitError(f"panel index not found: {p}")
        return PanelSource(json.loads(p.read_text(encoding="utf-8")), str(p.parent), True)

    url = index or site_base() + PANELS_INDEX_PATH
    cdir = (cache_dir or default_cache_dir()) / "panels" / _cache_name(url)
    dest = cdir / "index.json"
    notes: list[str] = []
    if offline:
        if not dest.exists():
            raise KitError(f"--offline given but no cached index at {dest}; run once online first")
        notes.append(f"offline: using cached index from {datetime.fromtimestamp(dest.stat().st_mtime, timezone.utc):%Y-%m-%d %H:%M} UTC")
    elif refresh or not _fresh(dest, INDEX_MAX_AGE_S):
        try:
            _download(url, dest)
        except (KitError, urllib.error.URLError, OSError) as e:
            if not dest.exists():
                raise KitError(f"could not download the panel index from {url}: {e}") from e
            notes.append(f"could not refresh the panel index ({e}); using the cached copy at {dest}")
    return PanelSource(json.loads(dest.read_text(encoding="utf-8")), url, False, notes)


def resolve_board(index: dict[str, Any], key: str) -> tuple[str, dict[str, Any]]:
    """Return (canonical key, contract). Matching ignores case."""
    if key in index:
        return key, index[key]
    lower = {k.lower(): k for k in index}
    if key.lower() in lower:
        k = lower[key.lower()]
        return k, index[k]
    raise KitError(f"unknown board {key!r}. Boards in the index: {', '.join(sorted(index))}")


def load_genes(
    source: PanelSource,
    board: dict[str, Any],
    *,
    cache_dir: Path | None = None,
    refresh: bool = False,
    offline: bool = False,
) -> tuple[list[str], str]:
    """Return (gene list, verification note) for one board contract."""
    name = board.get("genes_file")
    if not name:
        raise KitError("board contract has no genes_file")
    if source.is_local:
        path = Path(source.location) / name
        if not path.is_file():
            raise KitError(f"gene file not found next to the index: {path}")
        genes = _read_genes(path)
        return genes, _verify_genes(genes, board)

    url = urllib.parse.urljoin(source.location, name)
    dest = (cache_dir or default_cache_dir()) / "panels" / _cache_name(source.location) / name
    if not offline and (refresh or not dest.exists()):
        _download(url, dest)
    elif offline and not dest.exists():
        raise KitError(f"--offline given but gene file {name} is not cached")
    genes = _read_genes(dest)
    note = _verify_genes(genes, board)
    if note.startswith("MISMATCH") and not offline and not refresh:
        _download(url, dest)  # cached file may predate an index update
        genes = _read_genes(dest)
        note = _verify_genes(genes, board)
    return genes, note


def _read_genes(path: Path) -> list[str]:
    return [g.strip() for g in path.read_text(encoding="utf-8").splitlines() if g.strip()]


def _verify_genes(genes: list[str], board: dict[str, Any]) -> str:
    want = str(board.get("genes_sha256") or "").lower()
    n_want = board.get("n_genes")
    if n_want is not None and len(genes) != int(n_want):
        return f"MISMATCH: gene file has {len(genes)} names, index says n_genes={n_want}"
    if not want:
        return "no genes_sha256 in the index; gene file not verified"
    if genes_digest(genes).startswith(want):
        return "verified against genes_sha256"
    return "MISMATCH: gene file does not match genes_sha256 in the index"


# --------------------------------------------------------------------- board keys


def board_form_fields(board_key: str) -> dict[str, str]:
    """Upload form fields for a board key, mirroring the official web uploader.

    T1:<split>                  -> task=T1
    T2:<setting>:<split>_<mode> -> task=T2, setting=<setting>, mode=<mode>
    T3:<target>                 -> task=T3, target=<target>
    """
    parts = board_key.split(":")
    task = parts[0].upper()
    if task == "T1":
        return {"task": "T1"}
    if task == "T2":
        if len(parts) != 3 or "_" not in parts[2]:
            raise KitError(f"T2 board keys look like T2:heart:val_extrap, got {board_key!r}")
        return {"task": "T2", "setting": parts[1], "mode": parts[2].split("_")[1]}
    if task == "T3":
        return {"task": "T3", "target": parts[1]} if len(parts) > 1 and parts[1] else {"task": "T3"}
    raise KitError(f"board key must start with T1, T2 or T3, got {board_key!r}")


# ----------------------------------------------------------------------- API keys

_KEY_OK = re.compile(r"^[\x21-\x7e]+$")  # printable ASCII, no spaces or newlines


def read_api_key(key_file: str | None) -> tuple[str, str]:
    """Return (key, where it came from). The key itself is never printed or logged.

    Order: --key-file if given, else the VE_CHALLENGE_API_KEY environment variable.
    A key file may hold the bare key, or a line like `VE_CHALLENGE_API_KEY=...`
    (an optional leading `export` is accepted). Lines starting with # are ignored.
    """
    if key_file:
        p = Path(key_file).expanduser()
        if not p.is_file():
            raise KitError(f"key file not found: {p}")
        mode = p.stat().st_mode
        if mode & (stat.S_IRWXG | stat.S_IRWXO):
            print(f"warning: {p} is readable by other users; consider chmod 600", file=sys.stderr)
        key = ""
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" in line and line.split("=", 1)[0].strip().isidentifier():
                line = line.split("=", 1)[1].strip().strip("'\"")
            key = line
            break
        source = f"key file {p}"
    else:
        key = os.environ.get(API_KEY_ENV, "").strip()
        source = f"environment variable {API_KEY_ENV}"
    if not key:
        raise KitError(f"no API key: pass --key-file PATH or set {API_KEY_ENV}")
    if not _KEY_OK.match(key):
        raise KitError(f"the API key from {source} contains spaces or control characters")
    return key, source


def key_available(key_file: str | None) -> str:
    """Describe whether a key would be found, without reading it into output."""
    if key_file:
        return f"key file {key_file} ({'exists' if Path(key_file).expanduser().is_file() else 'NOT FOUND'})"
    return f"env {API_KEY_ENV} ({'set' if os.environ.get(API_KEY_ENV) else 'NOT SET'})"
