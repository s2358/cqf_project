# Task: Move pair rationale out of config.py, into the notebook

## Why
`config.PAIRS` currently has a `"rationale"` string field per pair. The
economic rationale for each pair should live as written prose in the
notebook, next to that pair's actual results — `config.py` should stay
purely mechanical (id/name/tickers only).

## 1. `config.py`

Remove the `"rationale": "..."` key from every entry in `PAIRS` (both the
active `KO_PEP` entry and the two commented-out example entries, `EWA_EWC`
and `MAR_IHG`). Each entry should end up with just `id`, `name`, and
`tickers`. Also update the comment block above `PAIRS` to note that
rationale belongs in the notebook, not here — add a line like:

> The economic rationale for each pair belongs in the notebook (as prose,
> next to that pair's results and discussion) rather than as a data field
> here — config.py stays purely mechanical.

Do not remove `id`, `name`, or `tickers` from any entry.

## 2. `README.md`

- In the "Adding / editing pairs" section, update the example `PAIRS`
  code block to remove the `"rationale"` line.
- Update the sentence introducing that code block to say each dict has
  `id`, `name`, and `tickers` (drop "and `rationale`").
- After the code block, add a note that the economic rationale should be
  written in the notebook as prose in each pair's markdown section, next
  to its results.
- In the notebook structure list further down, change "Intro + list of
  pairs and their rationale" to "Intro + list of pairs, with each one's
  economic rationale written out in markdown".

## 3. `notebooks/pairs_trading_analysis.ipynb`

Restructure the "Per-pair analysis" section so each pair has its own
dedicated markdown heading with the rationale as prose, instead of a
generic templated block that reads `pair['rationale']` from config.

Specifically:

- Delete any code cell that prints `pair['rationale']` or iterates
  `config.PAIRS` printing rationale.
- For each pair currently in `config.PAIRS` (start with just `KO_PEP`,
  since that's the only active one), add a markdown cell formatted like:

  ```
  ## Pair 1: Coca-Cola vs PepsiCo (`KO_PEP`)

  **Rationale:** *(edit this)* Same-sector consumer staples large caps
  with structurally similar demand drivers and pricing power — a
  starting point for testing whether sector comovement translates into a
  stable cointegrating relationship.
  ```

  followed by a code cell:
  ```python
  pair_id = 'KO_PEP'
  r = results[pair_id]
  pair = r['pair']
  ```

  then continue with the existing Step 1 (price plot) / Step 2
  (stationarity) / Step 3 (EG test + spread plot) / discussion cells,
  unchanged in content, just placed under this pair's heading.

- After the discussion cell, keep an instructional markdown cell telling
  the reader to duplicate the whole "Pair 1" block (heading + rationale
  through discussion cell) once per additional pair, updating the
  heading, written rationale, and `pair_id` each time.

- Leave the "Pairs under study" intro section, the "Run all pairs" cell,
  and the "Cross-pair comparison" section at the end unchanged, except:
  in "Pairs under study", change the code cell that printed each pair's
  rationale to just list `id`, `tickers`, and `name` (no rationale, since
  it no longer exists in config).

## 4. Validate

After making these changes:
- Confirm `config.py` still imports cleanly (`python -c "import config"`)
  and no other file in `src/` or `main.py` references `pair['rationale']`
  or `p['rationale']` anywhere (`grep -rn "rationale" .` should only hit
  the new prose in the notebook's markdown cells and comments).
- Execute the notebook top to bottom (e.g. via
  `jupyter nbconvert --to notebook --execute --inplace notebooks/pairs_trading_analysis.ipynb`)
  and confirm zero cell errors, using whatever price data is currently
  cached in `data/`. If no cached CSV exists yet, set `REFRESH_DATA = True`
  in the notebook's first data cell for this one validation run.
- Clear notebook outputs afterward
  (`jupyter nbconvert --clear-output --inplace notebooks/pairs_trading_analysis.ipynb`)
  so the committed notebook doesn't carry stale run outputs.
