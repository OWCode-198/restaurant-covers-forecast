"""Importing a raw Toast Sales summary export.

The transformations here are the ones that silently corrupt a dataset if they
go wrong: an 8-digit date read as an epoch offset, a half-finished trading day
treated as history, and a missing day written as a zero. Each gets a test.
"""
from __future__ import annotations

import datetime as dt
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"

spec = importlib.util.spec_from_file_location(
    "import_toast_export", SCRIPTS / "import_toast_export.py"
)
imp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imp)


def write_export(folder: Path, rows, dining=None) -> Path:
    """Build a minimal export. `rows` is (yyyyMMdd, sales, orders, guests)."""
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        rows, columns=["yyyyMMdd", "Net sales", "Total orders", "Total guests"]
    ).to_csv(folder / "Sales by day.csv", index=False)
    if dining is not None:
        pd.DataFrame(dining, columns=["Dining option", "Orders"]).to_csv(
            folder / "Dining options summary.csv", index=False
        )
    return folder


def ymd(day: dt.date) -> int:
    return int(day.strftime("%Y%m%d"))


def test_eight_digit_dates_are_not_read_as_epoch_offsets(tmp_path):
    """Regression: pd.to_datetime(20250825) silently yields 1970-01-01."""
    folder = write_export(tmp_path / "exp", [(20250825, 202.0, 22, 22)])
    assert imp.read_daily(folder)["date"].iloc[0] == pd.Timestamp("2025-08-25")


def test_todays_partial_row_is_dropped(tmp_path, capsys):
    today = dt.date.today()
    folder = write_export(
        tmp_path / "exp",
        [
            (ymd(today - dt.timedelta(days=2)), 800.0, 50, 50),
            (ymd(today - dt.timedelta(days=1)), 820.0, 56, 56),
            (ymd(today), 300.0, 20, 20),  # still trading
        ],
    )
    out = tmp_path / "clean.csv"
    assert imp.main([str(folder), "--out", str(out)]) == 0

    written = pd.read_csv(out, parse_dates=["date"])
    assert len(written) == 2
    assert written["date"].max().date() == today - dt.timedelta(days=1)
    assert "still trading when exported" in capsys.readouterr().out


def test_everything_partial_is_an_error_not_an_empty_file(tmp_path):
    folder = write_export(tmp_path / "exp", [(ymd(dt.date.today()), 300.0, 20, 20)])
    with pytest.raises(SystemExit):
        imp.main([str(folder), "--out", str(tmp_path / "clean.csv")])


def test_missing_days_are_reported_as_gaps_not_written_as_zero(tmp_path, capsys):
    base = dt.date.today() - dt.timedelta(days=10)
    rows = [
        (ymd(base), 100.0, 10, 10),
        # base+1 absent -- closed, or simply never recorded.
        (ymd(base + dt.timedelta(days=2)), 120.0, 12, 12),
    ]
    out = tmp_path / "clean.csv"
    assert imp.main([str(write_export(tmp_path / "exp", rows)), "--out", str(out)]) == 0

    written = pd.read_csv(out, parse_dates=["date"])
    assert len(written) == 2  # the gap is absent, not a zero row
    assert 0 not in written["covers"].tolist()
    assert "1 gap(s), left as unknown rather than zero" in capsys.readouterr().out


def test_identical_guests_and_orders_are_called_out(tmp_path, capsys):
    base = dt.date.today() - dt.timedelta(days=5)
    rows = [(ymd(base + dt.timedelta(days=i)), 100.0, 10, 10) for i in range(3)]
    imp.main([str(write_export(tmp_path / "exp", rows)), "--out", str(tmp_path / "c.csv")])
    assert "TRANSACTIONS, not covers" in capsys.readouterr().out


def test_real_guest_counts_are_reported_as_such(tmp_path, capsys):
    base = dt.date.today() - dt.timedelta(days=5)
    rows = [(ymd(base + dt.timedelta(days=i)), 100.0, 10, 25) for i in range(3)]
    imp.main([str(write_export(tmp_path / "exp", rows)), "--out", str(tmp_path / "c.csv")])
    out = capsys.readouterr().out
    assert "guest counts look real" in out
    assert "2.50 guests/order" in out


def test_existing_output_is_not_clobbered_without_force(tmp_path):
    base = dt.date.today() - dt.timedelta(days=5)
    folder = write_export(tmp_path / "exp", [(ymd(base), 100.0, 10, 10)])
    out = tmp_path / "clean.csv"
    out.write_text("precious,data\n1,2\n")

    with pytest.raises(SystemExit):
        imp.main([str(folder), "--out", str(out)])
    assert out.read_text() == "precious,data\n1,2\n"

    assert imp.main([str(folder), "--out", str(out), "--force"]) == 0
    assert "covers" in out.read_text()


def test_a_folder_without_the_daily_sheet_explains_itself(tmp_path):
    empty = tmp_path / "exp"
    empty.mkdir()
    with pytest.raises(SystemExit, match="Sales by day.csv"):
        imp.read_daily(empty)


def test_dining_mix_is_summarised_when_present(tmp_path, capsys):
    base = dt.date.today() - dt.timedelta(days=5)
    folder = write_export(
        tmp_path / "exp",
        [(ymd(base), 100.0, 10, 10)],
        dining=[("Take Out", 80), ("Dine In", 20), ("Total", 100)],
    )
    imp.main([str(folder), "--out", str(tmp_path / "c.csv")])
    out = capsys.readouterr().out
    assert "dining mix:" in out
    # The "Total" row must not be counted as its own dining option.
    assert "Take Out 80%" in out
