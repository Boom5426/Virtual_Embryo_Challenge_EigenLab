# vec-community-kit

Small, dependency-light command-line tools for participants in the
[Virtual Embryo Challenge](https://virtualembryo.ai) (VEC):

| Tool | What it does | Network |
|---|---|---|
| `vec_check.py` | Checks a prediction `.h5ad` against a board's public contract: genes and order, cell limits, finite non-negative X, log-scale sanity, `obsm['spatial_3D']`. Clear PASS/FAIL report and exit code. | Downloads the public board contracts once, then works offline |
| `vec_submit.py` | Safe uploader: local check, free server format check, then a scored upload only with `--scored`. Retries never spend a second attempt. Polls until scored, keeps a JSONL ledger, prints your quota before and after. | Your API key |
| `vec_board.py` | Reads the public leaderboards: per-board tables with per-metric scores, overall tables per track (with a check of the API's totals), one team's position. Saves every response to a timestamped folder. | Public endpoints only |
| `examples/copy_last_baseline.py` | Turns a released training file into a valid floor-model prediction for one board ("copy the last observed stage"), for testing your pipeline end to end. | Public board contracts |
| [`README.zh.md`](README.zh.md) | A beginner walkthrough in Chinese, from registration to a first submission. | |

This is an unofficial community tool. It is not affiliated with the organisers and
contains no modelling method. Where this README and the site disagree, the site
(rules, task pages, API guide) is the authority.

## Install

Python 3.10 or newer.

```bash
git clone <this repository> vec-community-kit
cd vec-community-kit
python -m pip install -r requirements.txt     # numpy, scipy, anndata, h5py
python -m unittest discover -s tests          # optional: 33 offline tests
```

Optional: `pip install veckit`, the organisers' scorer package. When it is
importable, `vec_check.py` also runs veckit's own file loader and reports the result.

## First submission in 15 minutes

1. **Register and get a key.** Sign in on the site, register your team (or open a
   teammate's invitation link) and accept the current Challenge Rules and Terms of
   Use. Your team captain creates an API key on the account's API keys page. The
   secret is shown only once. Save it to a private file, without putting it on a
   command line:

   ```bash
   umask 077; cat > ~/.vec_key      # paste the key, press Enter, then Ctrl-D
   ```

2. **Download one training file** from the Data page (`/challenge/data`). Ask the
   example which stage each board copies:

   ```bash
   python examples/copy_last_baseline.py --board T2:heart:val_extrap --hint
   ```

3. **Build a floor-model file** (a pipeline test, not a competitive entry):

   ```bash
   python examples/copy_last_baseline.py --board T2:heart:val_extrap \
       --input E9.5.h5ad --out floor_heart_extrap.h5ad
   ```

4. **Check it locally:**

   ```bash
   python vec_check.py floor_heart_extrap.h5ad --board T2:heart:val_extrap
   ```

5. **Run the free server format check** (no attempt is spent):

   ```bash
   python vec_submit.py floor_heart_extrap.h5ad --board T2:heart:val_extrap \
       --model "copy-last test" --key-file ~/.vec_key
   ```

6. **Spend one scored attempt** only when you mean it, by adding `--scored`.
   Preview the exact requests first with `--dry-run`, which sends nothing.

## Boards

Board keys come from the public contract index,
`https://virtualembryo.ai/challenge/panels/index.json`. On 2026-10-21 (validation
phase) it listed:

| Board | Genes | Cells (min..max) | Coordinates | Floor model |
|---|---|---|---|---|
| `T1:val` | 32,285 | 1000..5118 | no | copy_last |
| `T2:heart:val_interp` | 500 | 1000..17616 | yes | copy_last |
| `T2:heart:val_extrap` | 500 | 1000..25179 | yes | copy_last |
| `T2:embryo:val_interp` | 498 | 583..5000 | yes | copy_last |
| `T3:gata4` | 500 | 1000..7449 | yes | wt_identity |

Boards change between phases. `python vec_check.py --list-boards` prints the
current index, and `GET /challenge/boards` lists the boards that accept uploads now.

## vec_check.py

```bash
python vec_check.py PRED.h5ad --board T3:gata4            # human-readable report
python vec_check.py PRED.h5ad --board T3:gata4 --json     # machine-readable
python vec_check.py PRED.h5ad --board T1:val --strict     # warnings also fail
python vec_check.py --list-boards
```

The index and gene files are cached under `~/.cache/vec-kit` (override with
`VEC_KIT_CACHE`), refreshed after 24 hours or with `--refresh`; `--offline` uses the
cache only. A local index (`--index path/to/index.json`, gene files next to it) also
works.

| Check | Level | Basis |
|---|---|---|
| `.h5ad` extension, non-empty, at most 1,200 MiB | FAIL | web uploader; API guide upload limit |
| readable as AnnData, 2-D numeric X | FAIL | task pages |
| `var_names` identical to the board's gene file, in order | FAIL | task pages ("checked element by element") |
| cell count within the board's `min_cells..max_cells`, inclusive | FAIL | task pages and index |
| X has no NaN or infinity | FAIL | "a single NaN or infinity anywhere in the matrix rejects the file" |
| X is non-negative | FAIL | task pages |
| `obsm['spatial_3D']` present, shape (n, 3 or more), first three columns finite, when the board needs coordinates | FAIL | task pages |
| `obs` / `obsm` keys listed as required in the index | FAIL | index |
| integer-valued X (looks like raw counts) or very large values | WARN | the site warns that counts pass the format checks and are then scored as if log-scaled |
| `expm1(X)` row sums: median, range and largest deviation from 1e4 | WARN if any row is off by more than `--cp10k-tol` (1%) | see the note below |
| all sampled cells identical, all cells at one position, empty rows | WARN | sanity |
| the gene file matches the index's `genes_sha256` | WARN | integrity of the cached download |
| veckit's own loader, if installed | FAIL / SKIP | official package |

Note on normalisation: the site says Task 1 values are log1p-normalised and Task 2
and 3 values are log-normalised, and that counts which are normalised but not
log-transformed are wrong too. It does not state a target sum. The row-sum report
lets you compare your file with how the released training files are scaled (run the
checker on one of them). It is a warning, never a failure; check the site if unsure.

The index's `genes_sha256` matched the first 16 hex characters of SHA-256 over the
newline-joined gene names for every board on 2026-10-21. That convention is observed,
not documented, so a mismatch is a warning.

Exit codes: `0` no failures, `1` at least one failure (or a warning with `--strict`),
`2` usage or setup problem.

A local PASS is not a guarantee. The server's free format check is the authority,
and passing even that "does not guarantee scorer acceptance" (API guide).

## vec_submit.py

```bash
export VE_CHALLENGE_API_KEY=...        # or: --key-file ~/.vec_key
python vec_submit.py PRED.h5ad --board T2:heart:val_extrap --model "my method"            # free format check only
python vec_submit.py PRED.h5ad --board T2:heart:val_extrap --model "my method" --scored   # spends one attempt
python vec_submit.py PRED.h5ad --board T3:gata4 --model "my method" --scored --dry-run    # print requests, send nothing
```

Steps, in order:

1. `vec_check` locally (skip with `--skip-local-check`). A failure stops here.
2. Reads `/challenge/phase` and `/challenge/boards` and stops if the board is not open.
3. Prints `GET /challenge/integration/quota`.
4. `POST /challenge/submissions` with `format_only=true` and no Idempotency-Key, as
   the API guide prescribes. No submission is created and no attempt is spent.
5. Only with `--scored`: the scored upload, with
   `Idempotency-Key: vkit-<first 48 hex of sha256(board, model, file sha256, tag)>`.
   The same board, model name and file bytes always give the same key, so a retry
   cannot spend a second attempt. Timeouts and 5xx responses are retried with the
   identical request after 2, 4 and 8 seconds. 4xx responses are never retried.
6. Polls `GET /challenge/submissions/{id}` every 20 s (the guide asks for 15 to 30 s)
   until `scored`, `rejected` or `cancelled`, then prints the headline, per-metric
   scores, rank, the server's SHA-256 check against your file, and the quota again.

Form fields follow the board key, the same way the site's own uploader builds them:
`T1:val` sends `task=T1`; `T2:heart:val_extrap` sends `task=T2, setting=heart,
mode=extrap`; `T3:gata4` sends `task=T3, target=gata4`. `--model`,
`--architecture`, `--agent-framework` and `--agent-model` add the optional fields.

Safety details:

* The key is read from `--key-file` or `VE_CHALLENGE_API_KEY`, sent only in the
  `Authorization` header, and never printed, logged or placed on a command line.
* Every request and response is appended to `./vec_ledger.jsonl` (`--ledger`). If the
  ledger already holds a scored upload with the same key, the tool polls that
  submission instead of uploading again (`--resend` replays it; the server then
  returns the original response).
* To deliberately score identical bytes again, change `--attempt-tag`. Changing the
  file name, model name or other fields while keeping the same key makes the server
  answer 409, which spends nothing.
* Agent Team entries start as `awaiting_evidence`. This tool does not upload evidence:
  it stops with exit code 3 and the submission id, so you can attach the evidence in
  the portal or through the evidence endpoint.

Exit codes: `0` passed or scored, `1` failed or rejected, `2` usage or setup problem,
`3` waiting on something outside the tool (evidence, review, or the polling time limit).

## vec_board.py

```bash
python vec_board.py overall                      # human and agent tables, totals checked
python vec_board.py overall --track agent --top 50 --boards
python vec_board.py task T2                      # one table per T2 board, per-metric columns
python vec_board.py task T3 --track human --team "Your Team"
python vec_board.py team "Your Team"
python vec_board.py snapshot                     # save every public endpoint
python vec_board.py --from vec_snapshots/20261021T093245Z overall   # replay a snapshot
```

Each live run saves the JSON it fetched to `vec_snapshots/<UTC time>/` (change with
`--out`, disable with `--no-save`). The `overall` view recomputes every row as
`T1 + mean of the T2 boards + T3`, with an unanswered board counting as zero, and
reports any row where the API's own total differs. On 2026-10-21 all 413 rows
(teams, partner entries and reference rows) matched.

Observed on 2026-10-21, not documented: a board leaderboard returned at most 200
rows, while `/challenge/overall` listed every team. `team` therefore takes board
bests from `/challenge/overall` and per-metric scores from the board tables when
the team is among the returned rows.

## Publicly documented rules worth knowing

Each item was checked against the site on 2026-10-21. Sources: the Rules, the task
pages, and the API guide on the account's API keys page.

**Quota**

* Limits are per team, per task, per UTC day: 20 scored submissions per day in P1
  (development), 8 in P2 (baselines released), and in P3 (final test phase) 2 official
  submissions per board for the whole phase. `GET /challenge/phase` shows the current
  phase and its `daily_quota`.
* Website and API uploads share one budget. More keys or platforms do not add attempts.
* A rejected file does not spend a scored attempt, but every upload counts toward a
  separate daily upload limit. Format checks spend neither.
* `429` means a daily limit or a final-phase board budget is used up. Daily limits
  reset at the UTC day boundary; final-phase budgets do not reset.

**Rejections and file format**

* A rejected prediction can still return HTTP 200. Read `status` and `problems`, not
  only the HTTP code.
* Files above 1,200 MiB are refused (`413`), for format checks too.
* A cell count outside the board's limits is rejected before scoring and does not
  spend a scored attempt. Within the limits the count is free.
* `var_names` must be exactly the board's genes in the released order. Task 1 is
  whole-transcriptome (32,285 genes); the site names assuming the 500-gene MERFISH
  panel as the most common Task 1 schema error.
* A single NaN or infinity rejects the file. Negative values are not accepted (usually
  a sign the matrix was centred or z-scored).
* Raw counts are non-negative and finite, so they pass every format check and are then
  scored as if they were log-scaled. Nothing warns you on the server side.
* Tasks 2 and 3 need `obsm['spatial_3D']` with shape (cells, 3 or more), finite. Only
  the first three columns are read, and any coordinate frame is accepted.
* `obs['celltype']` is optional and ignored: the scorer types every cell itself.

**Retries and errors**

* Use a unique Idempotency-Key (1 to 128 letters, digits or `._:-`) for each new
  prediction and keep the header, file bytes, file name and form fields identical for
  every retry. An identical retry returns the original response without spending
  another attempt. The same key with a different payload returns `409`.
* Back off on timeouts and 5xx (for example 2, 4, then 8 seconds). Do not retry 4xx
  responses unchanged.
* `409` can also mean that the Rules or Terms need accepting again (see
  `stale_consents` in `GET /challenge/me`) or that submissions are closed.
* `401`: missing, invalid, expired or revoked key. Keys expire after the duration
  chosen at creation, no later than the submission deadline, and a captain change
  invalidates ordinary team keys.

**Scoring and ranking**

* Every board is scored on a scale where the floor model sits at 50 and the attainable
  ceiling at 100.
* A team's position on a board is its best score there, not its latest.
* The overall standing is the sum over tasks of each task's mean across its boards; a
  board never attempted counts as zero. Ranking is within a team's own track.
* Final rankings come from the hidden test sets, not the development leaderboard.

## Tests

```bash
python -m unittest discover -s tests -v
```

33 tests build tiny synthetic `.h5ad` files (passing files and each failure class
above), run `vec_check.py` against a fake local index, drive `vec_submit.py` against a
local mock API (format check, scored upload with a forced 503 and retry, ledger replay
protection, key never printed), and check the totals logic of `vec_board.py`. No
network is used.

## License

MIT, see [LICENSE](LICENSE). The challenge data, rules and scoring belong to the
organisers; this kit downloads only the public board contracts and redistributes no
challenge data.
