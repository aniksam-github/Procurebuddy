"""Streamlit dashboard for threshold regression results."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "testing" / "runs"
LATEST_RESULTS_FILE = ROOT / "threshold_regression_latest_results.json"
LATEST_SUMMARY_FILE = ROOT / "threshold_regression_latest_summary.json"

VALID_MODES = ["Direct Purchase", "LPC", "LTE", "OTE"]
UNCERTAINTY_PATTERNS = (
    "committee ?",
    "t&pc ?",
    "to be confirmed",
    "tbc",
    "not sure",
    "uncertain",
    "maybe",
)


def find_latest_results_file() -> Path | None:
    if LATEST_RESULTS_FILE.exists():
        return LATEST_RESULTS_FILE
    candidates = sorted(RUNS_DIR.glob("*/results.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def find_latest_summary_file() -> Path | None:
    if LATEST_SUMMARY_FILE.exists():
        return LATEST_SUMMARY_FILE
    candidates = sorted(RUNS_DIR.glob("*/summary.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def extract_mode(response: str) -> str:
    lowered = f" {str(response).lower()} "
    quick_answer_mode = None
    if "applicable mode:" in lowered:
        quick_answer_mode = lowered.split("applicable mode:", 1)[1].splitlines()[0]
    source = quick_answer_mode or lowered
    if "direct purchase" in source:
        return "Direct Purchase"
    if "lpc" in source or "local purchase committee" in source:
        return "LPC"
    if "lte" in source or "limited tender" in source:
        return "LTE"
    if "ote" in source or "open tender" in source:
        return "OTE"
    return "Unknown"


def strip_fenced_code_blocks(text: str) -> str:
    import re

    return re.sub(r"```.*?```", "", text, flags=re.DOTALL)


def has_uncertain_output(text: str) -> bool:
    import re

    prose = strip_fenced_code_blocks(str(text or "")).lower()
    if any(pattern in prose for pattern in UNCERTAINTY_PATTERNS):
        return True
    return bool(re.search(r"\b(?:committee|t&pc|lpc|lte|ote)\s*\?\b", prose))


def classify_bug(response: str, error: str = "") -> str:
    response = str(response or "")
    if error:
        return "REQUEST_ERROR"
    if "GFR 2025" in response:
        return "WRONG_SOURCE"
    if has_uncertain_output(response):
        return "UNCERTAIN"
    if "Total steps: 1" in response:
        return "WEAK_PROCESS"
    if "Rs.\n" in response or "Rs \n" in response:
        return "BROKEN_TEXT"
    if "FINAL DECISION:" not in response:
        return "MISSING_FINAL_DECISION"
    return "OTHER"


@st.cache_data(show_spinner=False)
def load_results(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return pd.DataFrame(data)
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_summary(path_str: str) -> dict:
    path = Path(path_str)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    st.set_page_config(page_title="ProcureBuddy Evaluation Dashboard", layout="wide")
    st.title("ProcureBuddy Evaluation Dashboard")

    latest_results = find_latest_results_file()
    latest_summary = find_latest_summary_file()

    with st.sidebar:
        st.header("Data Source")
        default_results_value = str(latest_results) if latest_results else ""
        results_path = st.text_input("Results file", value=default_results_value, help="Point to a results.json or CSV file.")
        default_summary_value = str(latest_summary) if latest_summary else ""
        summary_path = st.text_input("Summary file", value=default_summary_value, help="Optional summary.json path.")
        st.caption("Tip: overnight runner writes root-level latest files automatically.")

    if not results_path:
        st.warning("Run the overnight regression first, or enter a results file path in the sidebar.")
        return

    try:
        df = load_results(results_path)
    except Exception as exc:
        st.error(f"Could not load results: {exc}")
        return

    summary = load_summary(summary_path) if summary_path else {}

    if df.empty:
        st.warning("The selected results file is empty.")
        return

    if "predicted_mode" not in df.columns:
        df["predicted_mode"] = df.get("response", "").apply(extract_mode)
    if "bug_type" not in df.columns:
        df["bug_type"] = [classify_bug(resp, err) for resp, err in zip(df.get("response", ""), df.get("error", ""))]
    if "quality_score" not in df.columns:
        df["quality_score"] = 0.0
    if "passed" not in df.columns:
        expected = df.get("expected_mode", "")
        predicted = df["predicted_mode"]
        df["passed"] = expected == predicted

    total = len(df)
    passed = int(df["passed"].sum())
    failed = total - passed
    accuracy = passed / total * 100 if total else 0.0

    metric_cols = st.columns(5)
    metric_cols[0].metric("Total Tests", total)
    metric_cols[1].metric("Passed", passed)
    metric_cols[2].metric("Failed", failed)
    metric_cols[3].metric("Accuracy %", f"{accuracy:.2f}")
    metric_cols[4].metric("Avg Quality", f"{pd.to_numeric(df['quality_score'], errors='coerce').fillna(0).mean():.3f}")

    if summary:
        with st.expander("Run Metadata", expanded=False):
            st.json(summary)

    st.subheader("Rule-wise Accuracy")
    if "expected_mode" in df.columns:
        rule_stats = df.groupby("expected_mode")["passed"].agg(["count", "sum"]).rename(columns={"count": "total", "sum": "correct"})
        rule_stats["accuracy"] = rule_stats["correct"] / rule_stats["total"] * 100.0
        st.dataframe(rule_stats.sort_index(), use_container_width=True)

    st.subheader("Confusion Matrix")
    if "expected_mode" in df.columns:
        matrix = pd.crosstab(df["expected_mode"], df["predicted_mode"])
        st.dataframe(matrix, use_container_width=True)

    chart_cols = st.columns(2)

    with chart_cols[0]:
        st.subheader("Bug Distribution")
        bug_counts = df["bug_type"].value_counts()
        st.bar_chart(bug_counts)
        st.dataframe(bug_counts.rename_axis("bug").reset_index(name="count"), use_container_width=True)

    with chart_cols[1]:
        st.subheader("Predicted Modes")
        predicted_counts = df["predicted_mode"].value_counts()
        st.bar_chart(predicted_counts)
        st.dataframe(predicted_counts.rename_axis("predicted_mode").reset_index(name="count"), use_container_width=True)

    st.subheader("Failed Cases")
    failed_df = df[df["passed"] == False].copy()  # noqa: E712
    st.dataframe(
        failed_df[
            [
                column
                for column in [
                    "id",
                    "question",
                    "amount_text",
                    "expected_mode",
                    "predicted_mode",
                    "bug_type",
                    "validation_failures",
                    "error",
                ]
                if column in failed_df.columns
            ]
        ].head(100),
        use_container_width=True,
    )

    st.subheader("Search")
    search_text = st.text_input("Search by keyword")
    if search_text:
        searchable_columns = [column for column in ("question", "response", "bug_type", "expected_mode", "predicted_mode") if column in df.columns]
        mask = pd.Series(False, index=df.index)
        for column in searchable_columns:
            mask = mask | df[column].astype(str).str.contains(search_text, case=False, na=False)
        st.dataframe(df[mask], use_container_width=True)

    with st.expander("Raw Results", expanded=False):
        st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()
