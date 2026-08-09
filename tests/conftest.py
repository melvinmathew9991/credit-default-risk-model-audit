"""Shared test configuration: which dataset the suite runs against.

The full 22.8 MB dataset is not distributed with this repository. A stratified
sample is committed instead (see `analysis/make_sample.py`), chosen so that the
properties the tests assert on survive: the yearmo split, the default rate, the
`"0"` / `"0.0"` placeholder spelling that is finding H1, repeat customers, and
the `received_principal` leakage relationship.

Tests run against the full file when it is present and the sample otherwise, so
the suite is green either way.

One property cannot survive sampling. Finding H2 - identical text parsed as
different Python types depending on the csv chunk it lands in - needs more than
128k rows to cross a pandas chunk boundary. Tests for it are marked
`@pytest.mark.requires_full_dataset` and skip on the sample, because passing
them on 11k rows would mean the check had silently stopped working.
"""

import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FULL_DATA = os.path.join(ROOT, "data", "credit_risk_data.csv")
SAMPLE_DATA = os.path.join(ROOT, "data", "credit_risk_data_sample.csv")

#: Rows needed before pandas' chunked reader uses more than one chunk.
PANDAS_CHUNK_ROWS = 131_072


def resolve_data_path():
    """The dataset the suite should use: the full file if present, else the sample."""
    if os.path.exists(FULL_DATA):
        return FULL_DATA
    if os.path.exists(SAMPLE_DATA):
        return SAMPLE_DATA
    return None


DATA_PATH = resolve_data_path()
USING_FULL = DATA_PATH == FULL_DATA


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_full_dataset: needs the full dataset, not the committed sample",
    )


def pytest_collection_modifyitems(config, items):
    if USING_FULL:
        return
    skip = pytest.mark.skip(
        reason=(
            "requires the full dataset; running against the committed sample. "
            "See docs/DATA.md for how to obtain it."
        )
    )
    for item in items:
        if "requires_full_dataset" in item.keywords:
            item.add_marker(skip)


def pytest_report_header(config):
    if DATA_PATH is None:
        return "dataset: NONE FOUND - data-dependent tests will skip"
    kind = "full" if USING_FULL else "sample"
    return f"dataset: {kind} ({os.path.basename(DATA_PATH)})"


@pytest.fixture(scope="session")
def data_path():
    """Path to whichever dataset is available, skipping if there is none."""
    if DATA_PATH is None:
        pytest.skip("no dataset present; see docs/DATA.md")
    return DATA_PATH
