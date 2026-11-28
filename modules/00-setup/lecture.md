# Module 0 — Setup and tooling

In this module you set up a project that anyone (future you, or an interviewer) can clone and re-run to get exactly the same numbers. By the end, the tests pass locally and on GitHub, and a notebook can import the project's own Python package.

Notebook: [`notebook.ipynb`](notebook.ipynb). Code: [`statarb/config.py`](../../src/statarb/config.py).

> [!TIP]
> **How to read these notes.** The numbered steps are the main path: each does one thing, shows the result, and says what it means. The boxes are optional. Notes, tips and warnings are short asides; collapsed sections (click to expand) hold derivations and interview-level detail.

## Step 1. Create the environment

```bash
uv sync
```

`uv` reads `pyproject.toml` (the packages the project needs) and `uv.lock` (the exact version of every package, including indirect ones), and installs them into a private folder, `.venv/`.

**What it means:** everyone who runs `uv sync` gets the same versions of NumPy, pandas, yfinance and the rest, so results can't change because a library quietly changed underneath you.

> [!NOTE]
> A **virtual environment** is an isolated Python installation for one project. A **lockfile** (`uv.lock`) records the exact versions it was tested with.

## Step 2. Run the tests

```bash
uv run pytest
```

Result: a line such as `28 passed, 1 deselected`.

**What it means:** each function in `statarb` has a **unit test**, a small automatic check on a case where the right answer is known. The tests use **synthetic data** (simulated prices), never downloads; the one deselected test needs the internet and runs only on request.

<details>
<summary>Why test on synthetic data first: calibrating the instrument</summary>

Every estimator in this project is first run on simulated data where the true answer is known, the way you would calibrate an instrument before measuring. If an estimator can't recover a known parameter from a simulation, its output on market data means nothing. Synthetic tests are also fast and deterministic (they use fixed **random seeds**).

</details>

## Step 3. Lint and format the code

```bash
uv run ruff check
uv run ruff format
```

Result: `All checks passed!`, and formatting differences are fixed in place.

**What it means:** a **linter** flags suspicious code without running it (unused variables, shadowed names); the formatter ends all debate about layout.

## Step 4. Open a notebook that imports the package

```bash
uv run jupyter lab
```

Open `modules/00-setup/notebook.ipynb` and run it. The first cell prints the `statarb` version and the project's constants.

**What it means:** notebooks import tested code from `statarb` instead of copy-pasting it. Each notebook exists twice: `notebook.py` is the plain-text source, readable in a code review, and `notebook.ipynb` is the executed copy with outputs, so GitHub shows the plots.

## Step 5. Download one ticker and cache it

The notebook downloads ten years of SPY prices from Yahoo Finance, saves them under `data/`, and loads them twice:

```
load 1: download  0.470 s, 2516 rows
load 2: cache     0.014 s, 2516 rows
```

**What it means:** the second load reads a local file instead of the internet. Module 1 turns this into `statarb.data.load_prices`, which also records *when* each file was downloaded, because Yahoo revises its history.

> [!WARNING]
> Market data is **never committed** to the repository: Yahoo's terms don't allow redistribution, and the automated checks must not depend on a flaky download.

## Step 6. Learn the project conventions

These live in `src/statarb/config.py`:

- **252 trading days per year** (a **trading day** is a day the exchange is open). Returns are **close-to-close**: from one day's closing price to the next.
- **Returns are decimals** (0.01 means 1%); **costs are in basis points**: 1 bp = 0.01% $= 10^{-4}$.
- **Seeds are explicit.** Every random function takes a `seed` argument.
- **A signal computed with data up to day $t$ is traded at day $t+1$.**

> [!WARNING]
> The last rule matters most. If a decision uses today's closing price, you can't also trade *at* today's close. Breaking this rule is **look-ahead bias**, the most common way to fake a profitable **backtest** (a simulation of a trading rule on historical data). Module 1 shows how to test for it automatically.

## Step 7. Let CI check every change

Push a commit to GitHub. **Continuous integration (CI)**, here GitHub Actions, re-runs the lint, the format check and the tests on a clean machine and shows a green tick or a red cross on the pull request.

**What it means:** "works on my machine" is checked on someone else's machine every time.

**Done when:** `uv run pytest` and `uv run ruff check` pass locally and on GitHub, and `uv run jupyter lab` opens a notebook that imports `statarb`.

<details>
<summary>Why all this machinery? Four ways a backtest stops being reproducible</summary>

- **Silent environment drift.** A library upgrade changes a default (yfinance changed `auto_adjust` between versions) and your returns change without any code change. *Fix:* the lockfile.
- **Data drift.** Yahoo revises its history. *Fix:* a local cache with a recorded download date.
- **Unseeded randomness.** Simulations that differ on each run make it impossible to tell a bug from noise. *Fix:* explicit seeds.
- **Hidden state in notebooks.** Cells run out of order produce results the saved notebook can't reproduce. *Fix:* re-execute notebooks top to bottom before committing them.

The physics analogy is a lab notebook plus a calibrated apparatus: the lockfile is the apparatus spec, the download date is the run log, and the seed fixes the Monte Carlo.

</details>

<details>
<summary>Interview questions (4), with key points</summary>

Write your own answer first, then open the key points.

**1. How do you make a research result reproducible? Name at least four things you control.**

<details>
<summary>Key points</summary>

Pinned package versions (lockfile); cached input data with a recorded download date; explicit random seeds; code under version control with tests; notebooks re-executed top to bottom; automated checks on a clean machine (CI).

</details>

**2. Why don't you commit market data to the repository?**

<details>
<summary>Key points</summary>

Vendor terms usually forbid redistribution, data files bloat the repository, and tests that depend on downloaded data are slow and flaky. Tests use synthetic data; real data is cached locally with its download date.

</details>

**3. Why is a signal computed at day $t$ traded at day $t+1$? What goes wrong otherwise?**

<details>
<summary>Key points</summary>

A signal that uses day $t$'s close can only be acted on after that close. Trading it at the same close uses information you didn't have at decision time (look-ahead bias), which can make a worthless rule look excellent.

</details>

**4. Why test estimators on synthetic data before real data?**

<details>
<summary>Key points</summary>

On simulated data the true parameters are known, so you can check that the estimator recovers them and see its bias and noise. On market data there is no ground truth, so a bug and a discovery look the same.

</details>

</details>

<details>
<summary>Glossary of terms in this module</summary>

| Term | Meaning |
|---|---|
| Virtual environment | An isolated Python installation for one project |
| Lockfile | The exact versions of every package the project was tested with (`uv.lock`) |
| Unit test | A small automatic check on a case with a known answer |
| Linter | A tool that flags suspicious code without running it |
| Continuous integration (CI) | A server that re-runs the checks on a clean machine on every change |
| Trading day | A day the exchange is open; about 252 a year in the US |
| Close-to-close | Returns measured between consecutive closing prices |
| Basis point (bp) | One hundredth of a percent, $10^{-4}$ |
| Backtest | A simulation of a trading rule on historical data |
| Look-ahead bias | Using information that wasn't available at decision time |

</details>
