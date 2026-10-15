
from pathlib import Path
import json

import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

st.set_page_config(
    page_title="Instamart Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    [data-testid="stMetric"] {
        background: var(--secondary-background-color);
        padding: 16px;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }

    .hero {
        padding: 28px;
        border-radius: 16px;
        background: linear-gradient(120deg, #12372A, #436850);
        color: white;
        margin-bottom: 22px;
    }

    .hero h1 {
        color: white;
        margin-bottom: 8px;
    }

    .hero p {
        color: #E6F0E8;
        margin-bottom: 0;
        font-size: 1.05rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

def output_path(filename):
    """Return a path to an existing output file."""
    path = OUTPUT_DIR / filename
    return path if path.is_file() else None


def load_csv(filename):
    """Read a CSV file without modifying it."""
    path = output_path(filename)

    if path is None:
        return None

    try:
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return None


def load_json(filename):
    """Read a standard JSON file."""
    path = output_path(filename)

    if path is None:
        return None

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, ValueError):
        return None


def load_jsonl(filename):
    """Read a JSON Lines file one record at a time."""
    path = output_path(filename)

    if path is None:
        return None

    records = []

    try:
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    records.append({
                        "_parse_error": f"Invalid JSON on line {line_number}",
                        "_raw_line": line.strip(),
                    })

        return records

    except OSError:
        return None


def show_dataframe(dataframe, message):
    """Display a dataframe or a useful empty-state message."""
    if dataframe is None:
        st.info(message)
    elif dataframe.empty:
        st.info("This file exists but contains no records.")
    else:
        st.dataframe(
            dataframe,
            use_container_width=True,
            hide_index=True,
        )


def show_json(data, filename):
    """Display a JSON object or array."""
    if data is None:
        st.info(f"`outputs/{filename}` is unavailable.")
    else:
        st.json(data)


def show_jsonl(records, filename):
    """Display JSONL records in a readable table."""
    if records is None:
        st.info(f"`outputs/{filename}` is unavailable.")
        return

    if not records:
        st.info("This JSONL file contains no records.")
        return

    st.dataframe(
        pd.json_normalize(records),
        use_container_width=True,
        hide_index=True,
    )


def format_number(value):
    """Format a number for a dashboard metric."""
    if value is None:
        return "N/A"

    return f"{value:,}"


# Load saved outputs once per Streamlit cache cycle.
@st.cache_data(show_spinner=False)
def get_data():
    return {
        "segments": load_csv("customer_segments.csv"),
        "profiles": load_csv("segment_profiles.csv"),
        "clustering": load_csv("clustering_evaluation.csv"),
        "retention": load_csv("retention_list.csv"),
        "triage": load_csv("triage_results.csv"),
        "policy_evaluation": load_csv("policy_evaluation.csv"),
        "triage_report": load_json("triage_report.json"),
        "policy_answer": load_json("policy_answer.json"),
        "scenarios": load_json("agent_scenarios.json"),
        "agent_runs": load_jsonl("agent_runs.jsonl"),
        "tool_logs": load_jsonl("agent_tool_log.jsonl"),
        "refund_ledger": load_jsonl("refund_ledger.jsonl"),
        "zero_overlap": load_csv("zero_overlap_evaluation.csv"),
    }


data = get_data()

segments = data["segments"]
profiles = data["profiles"]
clustering = data["clustering"]
retention = data["retention"]
triage = data["triage"]
policy_evaluation = data["policy_evaluation"]


# ============================================================
# CUSTOMER SEGMENT COLUMN DETECTION
# ============================================================

SEGMENT_COLUMN = None

if segments is not None:
    if "segment_name" in segments.columns:
        SEGMENT_COLUMN = "segment_name"
    elif "segment_id" in segments.columns:
        SEGMENT_COLUMN = "segment_id"
    elif "segment" in segments.columns:
        SEGMENT_COLUMN = "segment"


def count_segments():
    """Count unique customer groups using the actual CSV schema."""
    if segments is None or SEGMENT_COLUMN is None:
        return None

    return int(segments[SEGMENT_COLUMN].nunique())


def count_scenarios():
    scenarios = data["scenarios"]

    if isinstance(scenarios, list):
        return len(scenarios)

    if isinstance(scenarios, dict):
        for key in ("scenarios", "tests", "cases"):
            if isinstance(scenarios.get(key), list):
                return len(scenarios[key])

        return len(scenarios)

    return None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🛒 Instamart")
st.sidebar.caption("Intelligence & Operations Platform")

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Customer Intelligence",
        "Support Operations",
        "Policy Agent",
        "Project Files",
    ],
)

st.sidebar.divider()

if st.sidebar.button("Refresh dashboard data", use_container_width=True):
    get_data.clear()
    st.rerun()

st.sidebar.caption("Instamart Capstone Project")
st.sidebar.caption("Analytics • Automation • AI")


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown(
        """
        <div class="hero">
            <h1>Instamart Intelligence</h1>
            <p>
                A data-driven platform for customer segmentation,
                retention insights, support triage, and policy assistance.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Project at a glance")

    customer_count = len(segments) if segments is not None else None
    segment_count = count_segments()
    ticket_count = len(triage) if triage is not None else None
    scenario_count = count_scenarios()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Customers analyzed", format_number(customer_count))
    col2.metric("Customer segments", format_number(segment_count))
    col3.metric("Support tickets", format_number(ticket_count))
    col4.metric("Agent scenarios", format_number(scenario_count))

    st.subheader("Platform modules")

    left, right = st.columns(2)

    with left:
        st.markdown("#### 👥 Customer Intelligence")
        st.write(
            "Explore customer groups, purchasing patterns, "
            "segment profiles, and retention outputs."
        )

        st.markdown("#### 🎫 Support Operations")
        st.write(
            "Search support tickets, inspect triage results, "
            "and review the generated report."
        )

    with right:
        st.markdown("#### 🤖 Policy Agent")
        st.write(
            "Inspect saved policy answers, evaluation results, "
            "agent scenarios, tool logs, and refund records."
        )

        st.markdown("#### 📁 Project Files")
        st.write(
            "Check the availability of important output files "
            "included with the project."
        )

    st.divider()
    st.subheader("Customer segment distribution")

    if segments is not None and SEGMENT_COLUMN is not None:
        distribution = (
            segments[SEGMENT_COLUMN]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
            .rename_axis("Segment")
            .reset_index(name="Customers")
        )

        chart_col, table_col = st.columns([3, 2])

        with chart_col:
            st.bar_chart(
                distribution.set_index("Segment"),
                y="Customers",
            )

        with table_col:
            st.dataframe(
                distribution,
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info(
            "Customer segmentation data or its segment column "
            "is unavailable."
        )

    st.caption(
        "All metrics are calculated from saved project outputs. "
        "The dashboard does not execute refund or agent actions."
    )


# ============================================================
# CUSTOMER INTELLIGENCE
# ============================================================

elif page == "Customer Intelligence":

    st.title("👥 Customer Intelligence")
    st.write(
        "Analyze customer segments and inspect the saved "
        "segmentation and retention outputs."
    )

    if segments is not None and not segments.empty:

        st.subheader("Customer overview")

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric("Customers", f"{len(segments):,}")

        if "monetary" in segments.columns:
            total_monetary = pd.to_numeric(
                segments["monetary"], errors="coerce"
            ).sum()
            metric2.metric("Total monetary value", f"₹{total_monetary:,.2f}")
        else:
            metric2.metric("Total monetary value", "N/A")

        if "is_premium" in segments.columns:
            premium = (
                segments["is_premium"]
                .astype(str)
                .str.lower()
                .isin(["true", "1", "yes"])
                .sum()
            )
            metric3.metric("Premium customers", f"{premium:,}")
        else:
            metric3.metric("Premium customers", "N/A")

        st.subheader("Segment distribution")

        if SEGMENT_COLUMN is not None:

            distribution = (
                segments[SEGMENT_COLUMN]
                .fillna("Unknown")
                .astype(str)
                .value_counts()
                .rename_axis("Segment")
                .reset_index(name="Customers")
            )

            st.bar_chart(distribution.set_index("Segment"))
            st.dataframe(
                distribution,
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("Filter customer records")

            available_segments = sorted(
                segments[SEGMENT_COLUMN]
                .fillna("Unknown")
                .astype(str)
                .unique()
                .tolist()
            )

            selected_segments = st.multiselect(
                "Choose customer segments",
                options=available_segments,
                default=available_segments,
            )

            filtered = segments[
                segments[SEGMENT_COLUMN]
                .fillna("Unknown")
                .astype(str)
                .isin(selected_segments)
            ].copy()

        else:
            filtered = segments.copy()

        st.write(f"Showing {len(filtered):,} customer records.")

        st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download customer data as CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="customer_segments_filtered.csv",
            mime="text/csv",
        )

        if "recency" in filtered.columns:
            st.subheader("Customer recency")
            recency_values = pd.to_numeric(
                filtered["recency"], errors="coerce"
            ).dropna()

            if not recency_values.empty:
                st.bar_chart(
                    recency_values.value_counts()
                    .sort_index()
                    .rename_axis("Days since last activity")
                    .to_frame("Customers")
                )

        if "frequency" in filtered.columns:
            st.subheader("Purchase frequency")
            frequency_values = pd.to_numeric(
                filtered["frequency"], errors="coerce"
            ).dropna()

            if not frequency_values.empty:
                st.bar_chart(
                    frequency_values.value_counts()
                    .sort_index()
                    .rename_axis("Purchase frequency")
                    .to_frame("Customers")
                )

    else:
        st.info("Customer segmentation output is unavailable.")

    st.divider()
    st.subheader("Segment profiles")
    show_dataframe(
        profiles,
        "The segment profile CSV is unavailable.",
    )

    st.divider()
    st.subheader("Clustering evaluation")
    show_dataframe(
        clustering,
        "The clustering evaluation CSV is unavailable.",
    )

    st.divider()
    st.subheader("Retention list")
    show_dataframe(
        retention,
        "The retention list CSV is unavailable.",
    )


# ============================================================
# SUPPORT OPERATIONS
# ============================================================

elif page == "Support Operations":

    st.title("🎫 Support Operations")
    st.write(
        "Search and review existing support ticket triage results."
    )

    if triage is not None and not triage.empty:

        st.metric("Tickets in results", f"{len(triage):,}")

        search = st.text_input(
            "Search tickets",
            placeholder="Enter an order ID, ticket ID, or keyword",
        )

        filtered = triage.copy()

        if search.strip():
            search_mask = filtered.astype(str).apply(
                lambda column: column.str.contains(
                    search.strip(),
                    case=False,
                    regex=False,
                    na=False,
                )
            ).any(axis=1)

            filtered = filtered[search_mask]

        st.write(f"Matching tickets: {len(filtered):,}")

        st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download filtered ticket results",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="triage_results_filtered.csv",
            mime="text/csv",
        )

        text_columns = [
            column
            for column in filtered.columns
            if filtered[column].dtype == "object"
        ]

        if text_columns:
            chosen_column = st.selectbox(
                "Visualize ticket categories",
                options=text_columns,
            )

            category_counts = (
                filtered[chosen_column]
                .fillna("Unknown")
                .astype(str)
                .value_counts()
                .head(15)
                .rename_axis("Category")
                .reset_index(name="Tickets")
            )

            st.subheader(f"Distribution: {chosen_column}")
            st.bar_chart(category_counts.set_index("Category"))

    else:
        st.info("Ticket triage output is unavailable.")

    st.divider()
    st.subheader("Triage report")
    show_json(data["triage_report"], "triage_report.json")


# ============================================================
# POLICY AGENT
# ============================================================

elif page == "Policy Agent":

    st.title("🤖 Policy Agent")

    st.write(
        "Explore saved policy-agent results, evaluation scenarios, "
        "execution logs, and refund records."
    )

    st.info(
        "Read-only dashboard: opening this page does not execute "
        "agent tools, issue refunds, or change the refund ledger."
    )

    st.subheader("Saved policy answer")
    show_json(data["policy_answer"], "policy_answer.json")

    st.divider()
    st.subheader("Policy evaluation results")
    show_dataframe(
        policy_evaluation,
        "The policy evaluation CSV is unavailable.",
    )

    st.divider()
    st.subheader("Agent scenarios")
    show_json(data["scenarios"], "agent_scenarios.json")

    st.divider()
    st.subheader("Agent run history")
    show_jsonl(data["agent_runs"], "agent_runs.jsonl")

    st.divider()
    st.subheader("Agent tool execution log")
    show_jsonl(data["tool_logs"], "agent_tool_log.jsonl")

    st.divider()
    st.subheader("Refund ledger")
    show_jsonl(data["refund_ledger"], "refund_ledger.jsonl")

    st.divider()
    st.subheader("Zero-overlap evaluation")
    show_dataframe(
        data["zero_overlap"],
        "The zero-overlap evaluation CSV is unavailable.",
    )


# ============================================================
# PROJECT FILES
# ============================================================

elif page == "Project Files":

    st.title("📁 Project Files")

    st.write(
        "Check which expected output files are available "
        "in this project deployment."
    )

    expected_files = [
        "customer_segments.csv",
        "segment_profiles.csv",
        "clustering_evaluation.csv",
        "retention_list.csv",
        "segmentation_report.txt",
        "lapse_report.txt",
        "lapse_model.joblib",
        "triage_results.csv",
        "triage_report.json",
        "policy_answer.json",
        "policy_evaluation.csv",
        "agent_scenarios.json",
        "agent_runs.jsonl",
        "agent_tool_log.jsonl",
        "refund_ledger.jsonl",
        "zero_overlap_evaluation.csv",
    ]

    rows = []

    for filename in expected_files:
        path = output_path(filename)

        rows.append({
            "File": filename,
            "Status": "Available" if path else "Not found",
            "Size (KB)": (
                round(path.stat().st_size / 1024, 2)
                if path
                else None
            ),
        })

    status_df = pd.DataFrame(rows)

    st.dataframe(
        status_df,
        use_container_width=True,
        hide_index=True,
    )

    available_count = sum(
        row["Status"] == "Available" for row in rows
    )

    st.progress(available_count / len(expected_files))

    st.caption(
        f"{available_count} of {len(expected_files)} expected "
        "output files are available."
    )

    st.divider()
    st.subheader("Project information")

    st.write("**Application:** Instamart Intelligence")
    st.write("**Framework:** Streamlit")
    st.write("**Data source:** Existing capstone output files")
    st.write("**Mode:** Read-only analytics dashboard")


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Instamart Capstone • Streamlit Dashboard • "
    "Built using existing project outputs"
)
