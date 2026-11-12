# Module 0 — Setup and tooling

**Goal:** a project anyone (including future you, or an interviewer) can clone and re-run to get the same numbers.

## 1. Why reproducibility matters here

A backtest is an experiment on historical data. Like any experiment, its result is only worth something if it can be repeated. In quantitative finance the failure modes are specific:

- **Silent environment drift.** A library upgrade changes a default (yfinance changed `auto_adjust` between versions), and your returns change without any code change.
- **Data drift.** Yahoo revises its history. Re-downloading next year can give slightly different prices.
- **Unseeded randomness.** Simulations that give different answers each run make it impossible to tell a bug from noise.
- **Hidden state in notebooks.** Cells run out of order produce results that the saved notebook cannot reproduce.

The physics analogy is the lab notebook plus a calibrated, documented apparatus. The lockfile is the apparatus spec; the data download date is the run log; the seed fixes the Monte Carlo.

## 2. The tools and what each one buys you

| Tool | What it does | Why it matters |
|---|---|---|
| `uv` | Creates the virtual environment and resolves dependencies into `uv.lock` | Everyone gets *exactly* the same package versions (`uv sync --locked`) |
| `src/statarb/` | Reusable code in an installable package | Notebooks import tested code instead of copy-pasting it |
| `pytest` | Unit tests in `tests/` | Each estimator is checked on synthetic data where the true answer is known |
| `ruff` | Linter and formatter | Catches bugs (unused variables, shadowed names) and removes style debates |
| `jupytext` | Pairs every `.ipynb` with a plain `.py` file | Notebook diffs become readable in pull requests |
| GitHub Actions CI | Runs lint and tests on every push and pull request | "Works on my machine" is checked on a clean machine |
| `pre-commit` | Runs ruff and jupytext before each commit | Problems are caught before they reach CI |

Daily commands:

```bash
uv sync                    # install / update the environment from uv.lock
uv run pytest              # tests (network tests are skipped by default)
uv run ruff check          # lint
uv run ruff format         # format
uv run jupyter lab         # notebooks
uv run pre-commit install  # once per clone
```

## 3. Project conventions

These live in `src/statarb/config.py` and are used everywhere:

- **Daily close-to-close data, 252 trading days per year.** US exchanges are closed at weekends and on about 9 holidays, leaving about 252 trading days.
- **Returns as decimals** (0.01 means 1%). **Costs in basis points:** 1 bp = 0.01% $= 10^{-4}$.
- **A signal computed with data up to day $t$ is traded at day $t+1$.** If you decide using today's closing price, you cannot also trade *at* today's close. Breaking this rule is called **look-ahead bias** and is the single most common way to fake a profitable backtest.
- **Seeds are explicit.** Every random function takes a `seed` argument.

## 4. Data policy

- Prices are downloaded from Yahoo Finance via `yfinance` and cached as parquet under `data/` (git-ignored).
- They are **never committed**: Yahoo's terms forbid redistribution, and CI must not depend on a flaky download.
- CI and unit tests use **synthetic data only**. Any test that needs the internet is marked `@pytest.mark.network` and skipped by default.
- Each cached download records its download date, because history gets revised.

## 5. Repository layout

```
modules/NN-topic/
    notebook.py     # source of truth (jupytext percent format)
    notebook.ipynb  # executed copy with outputs, so GitHub shows the plots
    lecture.md      # these notes: maths, finance context, glossary, interview questions
src/statarb/        # the shared package
tests/              # one test file per package module
GLOSSARY.md         # cumulative glossary across modules
```

## Glossary

- **Virtual environment:** an isolated Python installation for one project, so its packages don't clash with other projects.
- **Lockfile (`uv.lock`):** the exact version of every package (including indirect dependencies) that the project was tested with.
- **Unit test:** a small automatic check that one function gives the right answer on a case where you know the answer.
- **Continuous integration (CI):** a server that re-runs the checks on a clean machine on every change.
- **Linter:** a tool that flags suspicious or inconsistent code without running it.
- **Parquet:** a compressed, column-oriented file format for tables; fast to read and preserves data types.
- **Trading day:** a day the exchange is open.
- **Close-to-close:** returns measured from one day's closing price to the next day's.
- **Basis point (bp):** one hundredth of a percent, $10^{-4}$.
- **Look-ahead bias:** using information in a backtest that would not have been available at the time of the decision.
- **Backtest:** a simulation of how a trading rule would have performed on historical data.

## Interview questions

Answer these in your own words below each one.

1. How do you make a research result reproducible? Name at least four things you control.
2. Why don't you commit market data to the repository?
3. Why is a signal computed at day $t$ traded at day $t+1$? What goes wrong otherwise?
4. Why test estimators on synthetic data before real data?
