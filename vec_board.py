#!/usr/bin/env python3
"""vec_board: read the public VEC leaderboards from the terminal.

Uses only the unauthenticated endpoints of https://api.virtualembryo.ai:
/challenge/phase, /challenge/boards, /challenge/documents, /challenge/overall and
/challenge/leaderboard. Every live run saves what it fetched to a timestamped
folder (default ./vec_snapshots/<UTC time>/), so a table can be reproduced later
with --from.

Subcommands
  snapshot             save every public endpoint (overall per track, each board)
  overall              overall table per track; checks the API's totals against
                       total = T1 + mean of the T2 boards + T3 (unanswered board = 0)
  task T1|T2|T3        one table per board: headline and per-metric scores
  team NAME            one team's overall position and its row on each board

Scores are on the site's scale: the floor model scores 50 and the attainable
ceiling 100. A team's board score is its best on that board, and ranks are
within a track.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
from pathlib import Path
from typing import Any

import vec_common as vc

TRACKS = ("human", "agent")
TOL = 1e-6
POLITE_DELAY_S = 0.3


# ---------------------------------------------------------------- data source


class Source:
    """Fetch from the live API (saving each response) or read a saved snapshot."""

    def __init__(self, from_dir: Path | None, out_root: Path | None):
        self.from_dir = from_dir
        self.out_dir: Path | None = None
        if from_dir is None and out_root is not None:
            self.out_dir = out_root / vc.utc_stamp()
        self.meta: dict[str, Any] = {"api": vc.api_base(), "fetched": {}}

    @staticmethod
    def name_for(path: str, params: dict[str, str] | None) -> str:
        stem = path.strip("/").split("/")[-1]
        if params:
            stem += "__" + "__".join(f"{k}-{v}" for k, v in sorted(params.items()))
        return re.sub(r"[^A-Za-z0-9_.-]", "-", stem) + ".json"

    def get(self, path: str, params: dict[str, str] | None = None) -> Any:
        name = self.name_for(path, params)
        if self.from_dir is not None:
            f = self.from_dir / name
            if not f.is_file():
                raise vc.KitError(f"{f} is not in the snapshot; run without --from to fetch it")
            return json.loads(f.read_text(encoding="utf-8"))
        data = vc.get_public_json(path, params)
        time.sleep(POLITE_DELAY_S)
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            (self.out_dir / name).write_text(json.dumps(data, indent=1), encoding="utf-8")
            self.meta["fetched"][name] = {"path": path, "params": params or {}, "at": vc.utc_now()}
            (self.out_dir / "meta.json").write_text(json.dumps(self.meta, indent=1), encoding="utf-8")
        return data


# ------------------------------------------------------------------ helpers


def _fmt(x: Any, width: int = 6) -> str:
    if x is None:
        return "-".rjust(width)
    try:
        return f"{float(x):{width}.1f}"
    except (TypeError, ValueError):
        return str(x)[:width].rjust(width)


def _cut(s: Any, n: int) -> str:
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[: n - 1] + "~"


def task_of(boards: list[dict[str, Any]]) -> dict[str, str]:
    return {b["key"]: b.get("task") or b["key"].split(":")[0] for b in boards}


def recompute(row: dict[str, Any], board_task: dict[str, str]) -> tuple[dict[str, float], float]:
    """Task score = mean over the task's boards (missing board = 0); total = sum of tasks."""
    tasks: dict[str, float] = {}
    for t in sorted(set(board_task.values())):
        keys = [k for k, v in board_task.items() if v == t]
        vals = [float((row.get("boards") or {}).get(k) or 0.0) for k in keys]
        tasks[t] = sum(vals) / len(keys)
    return tasks, sum(tasks.values())


def verify_totals(overall: dict[str, Any]) -> tuple[int, list[str]]:
    board_task = task_of(overall.get("boards") or [])
    rows = (overall.get("rows") or []) + (overall.get("partners") or []) + (overall.get("reference") or [])
    bad = []
    for r in rows:
        tasks, total = recompute(r, board_task)
        api_tasks = r.get("tasks") or {}
        if abs(total - float(r.get("total") or 0.0)) > TOL or any(
                abs(tasks[t] - float(api_tasks.get(t) or 0.0)) > TOL for t in tasks):
            bad.append(f"{r.get('team')}: API total {r.get('total')}, recomputed {total:.4f}")
    return len(rows), bad


def find_team(rows: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    exact = [r for r in rows if str(r.get("team", "")).lower() == name.lower()]
    if exact:
        return exact
    return [r for r in rows if name.lower() in str(r.get("team", "")).lower()]


# ------------------------------------------------------------------- views


def print_overall(src: Source, track: str, top: int, team: str | None, show_boards: bool) -> None:
    phase = src.get("/challenge/phase")
    overall = src.get("/challenge/overall")
    boards = overall.get("boards") or []
    print(f"Overall standings  phase {phase.get('label')}  split {overall.get('split')}  updated {overall.get('updated_at')}")
    n, bad = verify_totals(overall)
    if bad:
        print(f"TOTAL CHECK: {len(bad)} of {n} rows disagree with T1 + mean(T2 boards) + T3:")
        for b in bad[:10]:
            print(f"  {b}")
    else:
        print(f"Total check: all {n} rows (teams, partners, references) satisfy total = T1 + mean(T2 boards) + T3")
    keys = [b["key"] for b in boards]
    for tr in (TRACKS if track == "all" else (track,)):
        rows = [r for r in overall.get("rows") or [] if r.get("track") == tr]
        print(f"\n{tr.capitalize()} Team track: {len(rows)} teams")
        head = f"{'rank':>4}  {'team':<26} {'T1':>6} {'T2':>6} {'T3':>6} {'total':>7}"
        if show_boards:
            head += "  " + " ".join(f"{_cut(k, 12):>12}" for k in keys)
        print(head)
        shown = rows[:top]
        if team:
            shown += [r for r in find_team(rows, team) if r not in shown]
        for r in shown:
            t = r.get("tasks") or {}
            line = (f"{r.get('rank') or '-':>4}  {_cut(r.get('team'), 26):<26} {_fmt(t.get('T1'))} {_fmt(t.get('T2'))} "
                    f"{_fmt(t.get('T3'))} {_fmt(r.get('total'), 7)}")
            if show_boards:
                line += "  " + " ".join(_fmt((r.get("boards") or {}).get(k), 12) for k in keys)
            print(line)
    refs = overall.get("reference") or []
    if refs:
        print("\nReference entries (not ranked):")
        for r in refs:
            floor = "  [floor]" if r.get("is_floor") else ""
            print(f"      {_cut(r.get('team'), 26):<26} total {_fmt(r.get('total'), 7)}{floor}")
    partners = overall.get("partners") or []
    if partners:
        print("\nPartner reference entries (no official rank or prize eligibility):")
        for r in partners:
            print(f"      {_cut(r.get('team'), 26):<26} total {_fmt(r.get('total'), 7)}")


def print_task(src: Source, task: str, board: str | None, track: str, top: int, team: str | None) -> None:
    phase = src.get("/challenge/phase")
    boards = [b for b in (src.get("/challenge/boards").get("boards") or []) if b.get("task") == task]
    if board:
        boards = [b for b in boards if b["key"].lower() == board.lower()]
        if not boards:
            raise vc.KitError(f"{board} is not an open {task} board now")
    if not boards:
        raise vc.KitError(f"no open boards for {task}")
    print(f"{task} leaderboards  phase {phase.get('label')}")
    for b in boards:
        params = {"task": task, "board": b["key"]}
        if track != "all":
            params["track"] = track
        lb = src.get("/challenge/leaderboard", params)
        rows = lb.get("rows") or []
        metric_keys: list[str] = []
        for r in rows + (lb.get("reference") or []):
            for k in (r.get("metrics") or {}):
                if k not in metric_keys:
                    metric_keys.append(k)
        print(f"\n{b['key']}  ({b.get('label')})  track {track}  rows returned {len(rows)}  "
              f"counts {json.dumps(lb.get('counts'))}  shows_scores {lb.get('shows_scores')}")
        if lb.get("note"):
            print(f"note: {lb['note']}")
        head = f"{'rank':>4}  {'team':<24} {'track':<5} {'model':<14} {'head':>6}  " + " ".join(
            f"{vc.METRIC_ABBR.get(k, k)[:6]:>6}" for k in metric_keys)
        print(head)
        shown = rows[:top]
        if team:
            shown += [r for r in find_team(rows, team) if r not in shown]
        for r in shown:
            m = r.get("metrics") or {}
            print(f"{r.get('rank') or '-':>4}  {_cut(r.get('team'), 24):<24} {_cut(r.get('track'), 5):<5} "
                  f"{_cut(r.get('model'), 14):<14} {_fmt(r.get('headline'))}  " + " ".join(_fmt(m.get(k)) for k in metric_keys))
        for r in lb.get("reference") or []:
            m = r.get("metrics") or {}
            print(f"{'ref':>4}  {_cut(r.get('name'), 24):<24} {'':<5} {'':<14} {_fmt(r.get('headline'))}  "
                  + " ".join(_fmt(m.get(k)) for k in metric_keys))
        for r in lb.get("partners") or []:
            print(f"{'ptnr':>4}  {_cut(r.get('team'), 24):<24} {_cut(r.get('track'), 5):<5} {'':<14} {_fmt(r.get('headline'))}")
        if team and not find_team(rows, team):
            print(f"  {team!r} is not among the {len(rows)} rows returned for this board (see `team` for its best score)")
    print("\nhead = headline. Later columns are per-metric scores; the floor and ceiling reference rows show their scale.")


def print_team(src: Source, name: str) -> None:
    overall = src.get("/challenge/overall")
    matches = find_team(overall.get("rows") or [], name)
    partner = find_team(overall.get("partners") or [], name)
    if not matches and not partner:
        raise vc.KitError(f"no team matching {name!r} in /challenge/overall")
    if len(matches) > 1 and all(r.get("team", "").lower() != name.lower() for r in matches):
        print("Several teams match; be more specific:")
        for r in matches[:20]:
            print(f"  {r.get('team')}")
        return
    board_task = task_of(overall.get("boards") or [])
    for r in matches + partner:
        tr = r.get("track")
        tasks, total = recompute(r, board_task)
        ok = abs(total - float(r.get("total") or 0)) <= TOL
        rank = r.get("rank")
        n_track = sum(1 for x in overall.get("rows") or [] if x.get("track") == tr)
        print(f"{r.get('team')}  ({r.get('affiliation') or 'no affiliation'})  track {tr}")
        print(f"  overall rank {rank if rank is not None else '- (not ranked)'} of {n_track} in the {tr} track, "
              f"total {_fmt(r.get('total'), 7).strip()} (recomputed {total:.4f}, {'matches' if ok else 'DIFFERS'})")
        print("  tasks: " + "  ".join(f"{t} {_fmt((r.get('tasks') or {}).get(t)).strip()}" for t in sorted(tasks)))
        for key, task in board_task.items():
            best = (r.get("boards") or {}).get(key)
            lb = src.get("/challenge/leaderboard", {"task": task, "board": key, "track": tr})
            row = next((x for x in lb.get("rows") or [] if str(x.get("team", "")).lower() == str(r.get("team")).lower()), None)
            if row:
                m = " ".join(f"{vc.METRIC_ABBR.get(k, k)} {_fmt(v).strip()}" for k, v in (row.get("metrics") or {}).items())
                print(f"  {key:<22} best {_fmt(best).strip():>5}  board rank {row.get('rank')} of {len(lb.get('rows') or [])} returned  {m}")
            else:
                shown = "not attempted (counts as 0)" if best is None else "not in the rows returned for this board"
                print(f"  {key:<22} best {_fmt(best).strip():>5}  {shown}")


def snapshot_all(src: Source) -> None:
    src.get("/challenge/phase")
    boards = src.get("/challenge/boards").get("boards") or []
    src.get("/challenge/documents")
    src.get("/challenge/overall")
    for tr in TRACKS:
        src.get("/challenge/overall", {"track": tr})
    for b in boards:
        src.get("/challenge/leaderboard", {"task": b["task"], "board": b["key"]})
        for tr in TRACKS:
            src.get("/challenge/leaderboard", {"task": b["task"], "board": b["key"], "track": tr})
    print(f"saved {len(src.meta['fetched'])} responses to {src.out_dir}")


# --------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vec_board.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("vec_snapshots"), help="snapshot root (default ./vec_snapshots)")
    p.add_argument("--no-save", action="store_true", help="do not save the fetched JSON")
    p.add_argument("--from", dest="from_dir", type=Path, help="read a saved snapshot folder instead of the live API")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("snapshot", help="save every public endpoint")
    o = sub.add_parser("overall", help="overall table per track")
    o.add_argument("--track", choices=("all",) + TRACKS, default="all")
    o.add_argument("--top", type=int, default=20)
    o.add_argument("--team", help="also show this team's row")
    o.add_argument("--boards", action="store_true", help="add one column per board")
    t = sub.add_parser("task", help="per-board tables for one task")
    t.add_argument("task", choices=("T1", "T2", "T3"))
    t.add_argument("--board", help="only this board key")
    t.add_argument("--track", choices=("all",) + TRACKS, default="all")
    t.add_argument("--top", type=int, default=20)
    t.add_argument("--team", help="also show this team's row")
    m = sub.add_parser("team", help="one team's position")
    m.add_argument("name")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    src = Source(args.from_dir, None if args.no_save else args.out)
    try:
        if args.cmd == "snapshot":
            if src.out_dir is None:
                raise vc.KitError("snapshot needs saving enabled and no --from")
            snapshot_all(src)
        elif args.cmd == "overall":
            print_overall(src, args.track, args.top, args.team, args.boards)
        elif args.cmd == "task":
            print_task(src, args.task, args.board, args.track, args.top, args.team)
        elif args.cmd == "team":
            print_team(src, args.name)
    except (vc.KitError, urllib.error.URLError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if src.out_dir is not None and src.meta["fetched"] and args.cmd != "snapshot":
        print(f"\n(saved {len(src.meta['fetched'])} responses to {src.out_dir})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
