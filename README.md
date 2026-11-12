# starb — learning statistical arbitrage in Python

[![CI](https://github.com/aaaalex-m/starb/actions/workflows/ci.yml/badge.svg)](https://github.com/aaaalex-m/starb/actions/workflows/ci.yml)

A hands-on study project in **statistical arbitrage** (stat arb): trading many small, statistically favourable bets that the relative prices of related assets will revert, while staying roughly neutral to the overall market. It is written by a physicist learning finance and econometrics from scratch, and aims at understanding rather than trading. Every line of code, every formula and every piece of jargon should be explainable in an interview.

Each module is three things:

1. **A notebook** (`modules/NN-topic/notebook.ipynb`, executed, with outputs) that simulates first, where the true answer is known, and then uses real market data.
2. **Lecture notes** (`modules/NN-topic/lecture.md`): the maths (LaTeX, rendered by GitHub), the finance context, a glossary of new jargon, and interview questions.
3. **Tested code** added to the `statarb` package (`src/statarb/`), with unit tests on synthetic data.

A word on expectations: simple pairs strategies on liquid US stocks have mostly stopped making money after costs since the 2000s. A careful, honest negative result is the goal, not a suspiciously good backtest.

## Roadmap

| # | Module | `statarb` code | Status |
|---|---|---|---|
| 0 | [Setup and tooling](modules/00-setup/lecture.md) | skeleton, `config`, CI | done |
| 1 | Markets, prices and returns | `data`, `returns` | next |
| 2 | Data pitfalls: survivorship and look-ahead bias | `universe`, `data.clean_prices` | |
| 3 | Statistics toolkit: random walks, stationarity, regression | `simulate`, `stats` | |
| 4 | Distance-method pairs trading (Gatev et al. 2006) | `pairs.distance`, `signals` | |
| 5 | Cointegration: Engle–Granger and ADF | `coint` | |
| 6 | Ornstein–Uhlenbeck process and half-life | `ou` | |
| 7 | Backtester and performance metrics | `backtest`, `metrics` | |
| 8 | Honest backtesting: costs, walk-forward, multiple testing | `costs`, `validation` | |
| 9 | Kalman filter hedge ratios | `kalman` | |
| 10 | Johansen test and VECM baskets | `johansen` | |
| 11 | Machine learning for pair selection | `selection`, `networks` | |
| 12 | Factor models and random matrix theory | `factors`, `rmt` | |
| 13 | Factor-residual stat arb (Avellaneda & Lee 2010) | `residual` | |
| 14 | Capstone: risk, limits of arbitrage, interview prep | `risk` | |

Optional extensions (regime HMMs, meta-labelling, autoencoder factors, RL thresholds, deep-learning stat arb) live in `statarb/extensions/` if and when they are done. The cumulative jargon list is in [GLOSSARY.md](GLOSSARY.md); references are in [references/references.bib](references/references.bib).

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
git clone https://github.com/aaaalex-m/starb && cd starb
uv sync                    # creates .venv from uv.lock, installs statarb in editable mode
uv run pytest              # unit tests (synthetic data; network tests skipped)
uv run ruff check          # lint
uv run jupyter lab         # open modules/*/notebook.ipynb
uv run pre-commit install  # optional: lint and sync notebooks before each commit
```

Notebooks are paired with percent-format `.py` files by [jupytext](https://jupytext.readthedocs.io/). Edit either one; `jupytext --sync modules/NN-topic/notebook.ipynb` (or the pre-commit hook) keeps them aligned. CI checks that they match.

## Conventions

Defined in [`src/statarb/config.py`](src/statarb/config.py):

- Daily close-to-close data; **252 trading days per year**.
- Returns are stored as **decimals** (0.01 = 1%); costs are quoted in **basis points** (1 bp = 0.01%).
- **A signal computed with data up to day *t* is traded at day *t*+1.** This one rule prevents most look-ahead bugs.
- Random seeds are passed explicitly everywhere.

## Data policy

Prices come from Yahoo Finance via `yfinance` and are cached as parquet under `data/`, which is git-ignored. Market data is **never committed**: Yahoo's terms don't allow redistribution, and CI must not depend on a flaky download. Unit tests and CI use synthetic data only. Each cached download records its download date, because Yahoo revises history.

## Layout

```
starb/
├── modules/NN-topic/   notebook.py (source), notebook.ipynb (executed), lecture.md
├── src/statarb/        the shared package, one file per topic
├── tests/              pytest, one file per package module
├── references/         references.bib
├── GLOSSARY.md         cumulative glossary
└── data/               git-ignored parquet cache (raw/, processed/)
```

## Licence

[MIT](LICENSE).
