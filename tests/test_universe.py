import pandas as pd
import pytest

from statarb import universe


@pytest.fixture
def tables():
    """A toy index: today {A, B, C}, with four historical changes."""
    current = pd.DataFrame({"ticker": ["A", "B", "C"]})
    changes = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-06-01", "2018-01-02", "2015-01-02", "2012-03-01"]),
            "added": ["C", "B", None, "D"],
            "removed": ["D", None, "E", "F"],
            "reason": ["", "", "", ""],
        }
    )
    return current, changes


@pytest.mark.parametrize(
    ("date", "expected"),
    [
        ("2021-01-01", ["A", "B", "C"]),
        ("2020-06-01", ["A", "B", "C"]),  # change effective that day is kept
        ("2020-05-29", ["A", "B", "D"]),
        ("2017-06-30", ["A", "D"]),
        ("2014-12-31", ["A", "D", "E"]),
        ("2010-01-01", ["A", "E", "F"]),
    ],
)
def test_members_asof_undoes_later_changes(tables, date, expected):
    current, changes = tables
    assert universe.members_asof(date, current["ticker"], changes) == expected


def test_sp500_members_asof_uses_given_tables(tables):
    current, changes = tables
    assert universe.sp500_members_asof("2017-06-30", current, changes) == ["A", "D"]


def test_membership_matrix_matches_members_asof(tables):
    current, changes = tables
    dates = pd.to_datetime(["2010-01-01", "2017-06-30", "2021-01-01"])
    matrix = universe.membership_matrix(dates, current["ticker"], changes)
    assert list(matrix.columns) == ["A", "B", "C", "D", "E", "F"]
    for d in dates:
        assert sorted(matrix.columns[matrix.loc[d]]) == universe.members_asof(
            d, current["ticker"], changes
        )


def test_to_yahoo_ticker():
    assert universe.to_yahoo_ticker(" brk.b ") == "BRK-B"
    assert universe.to_yahoo_ticker("KO") == "KO"


def test_parse_changes_table_handles_two_header_rows():
    columns = pd.MultiIndex.from_tuples(
        [
            ("Effective Date", "Effective Date"),
            ("Added", "Ticker"),
            ("Added", "Security"),
            ("Removed", "Ticker"),
            ("Removed", "Security"),
            ("Reason", "Reason"),
            ("Refs", "Refs"),
        ]
    )
    raw = pd.DataFrame(
        [
            ["October 6, 2026", "TWLO", "Twilio", "WBD", "Warner Bros.", "Acquired", "[3]"],
            ["October 6, 2026", None, None, "CTVA", "Corteva", "Market cap", "[3]"],
            ["July 1, 1976", "BRK.B", "Berkshire", None, None, "Reorganised", "[9]"],
            ["not a date", "X", "X", "Y", "Y", "", ""],
        ],
        columns=columns,
    )
    changes = universe.parse_changes_table(raw)
    assert len(changes) == 3
    assert changes.loc[0, "date"] == pd.Timestamp("2026-10-06")
    assert changes["added"].fillna("").tolist() == ["TWLO", "", "BRK-B"]
    assert changes["removed"].fillna("").tolist() == ["WBD", "CTVA", ""]


def test_parse_current_table():
    raw = pd.DataFrame(
        {
            "Symbol": ["MMM", "BRK.B"],
            "Security": ["3M", "Berkshire Hathaway"],
            "GICS Sector": ["Industrials", "Financials"],
            "GICS Sub-Industry": ["Conglomerates", "Insurance"],
            "Date added": ["1957-03-04", "2010-02-16"],
        }
    )
    current = universe.parse_current_table(raw)
    assert current["ticker"].tolist() == ["BRK-B", "MMM"]
    assert current.loc[0, "date_added"] == pd.Timestamp("2010-02-16")


def test_ticker_lists_are_consistent():
    assert len(universe.LARGE_CAP_100) == 100
    assert len(set(universe.LARGE_CAP_100)) == 100
    assert len(universe.U1) == len(set(universe.U1))
    assert {"SPY", *universe.SECTOR_ETFS} <= set(universe.U1)
    assert all(a in universe.U1 and b in universe.U1 for a, b in universe.CLASSIC_PAIRS)


@pytest.mark.network
def test_wikipedia_reconstruction_has_about_500_members(tmp_path):
    current, changes = universe.load_sp500_tables(cache_dir=tmp_path)
    assert 495 <= len(current) <= 510
    assert 480 <= len(universe.sp500_members_asof("2016-01-04", current, changes)) <= 520
