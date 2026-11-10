import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Service Requests", page_icon="🎫", layout="wide")
st.title("Service Requests Dashboard")
st.caption("Live view of confirmed requests in the local SQLite database")

DB_PATH = Path(__file__).resolve().parent.parent / "service_requests.db"

@st.cache_data(ttl=30)
def load_requests(db_path: str, modified_at: float) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(
            """SELECT id, ticket_number, compound, building, unit_number, name,
                      issue_description, category, urgency, status, created_at
               FROM service_requests ORDER BY created_at DESC""",
            conn,
        )

if not DB_PATH.exists():
    st.error(f"Database not found: {DB_PATH}")
    st.stop()

try:
    requests = load_requests(str(DB_PATH), DB_PATH.stat().st_mtime)
except Exception as exc:
    st.error(f"Could not read service_requests: {exc}")
    st.stop()

if requests.empty:
    st.info("No service requests have been recorded yet.")
    st.stop()

requests["created_at"] = pd.to_datetime(requests["created_at"], errors="coerce")
requests["day"] = requests["created_at"].dt.date
for col in ["compound", "building", "unit_number", "name", "category", "urgency", "status"]:
    requests[col] = requests[col].fillna("Unknown").astype(str)

with st.sidebar:
    st.header("Filters")
    min_day = requests["day"].min()
    max_day = requests["day"].max()
    date_range = st.date_input("Date range", value=(min_day, max_day), min_value=min_day, max_value=max_day)
    selected_compounds = st.multiselect("Compound", sorted(requests["compound"].unique()))
    selected_categories = st.multiselect("Category", sorted(requests["category"].unique()))
    selected_statuses = st.multiselect("Status", sorted(requests["status"].unique()))
    selected_urgencies = st.multiselect("Urgency", sorted(requests["urgency"].unique()))

filtered = requests.copy()
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    filtered = filtered[filtered["day"].between(date_range[0], date_range[1])]
for column, selected in [("compound", selected_compounds), ("category", selected_categories), ("status", selected_statuses), ("urgency", selected_urgencies)]:
    if selected:
        filtered = filtered[filtered[column].isin(selected)]

open_statuses = {"new", "open", "assigned", "in progress", "pending"}
open_count = filtered["status"].str.lower().isin(open_statuses).sum()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Requests", f"{len(filtered):,}")
col2.metric("Open / active", f"{open_count:,}")
col3.metric("Compounds", f"{filtered['compound'].nunique():,}")
col4.metric("Categories", f"{filtered['category'].nunique():,}")

left, right = st.columns(2)
with left:
    st.subheader("Requests by day")
    daily = filtered.groupby("day").size().rename("Requests")
    st.line_chart(daily)
    st.subheader("By category")
    category_counts = filtered["category"].value_counts().rename_axis("Category").reset_index(name="Requests")
    st.vega_lite_chart(
        category_counts,
        {
            "mark": {"type": "arc", "innerRadius": 45},
            "encoding": {
                "theta": {"field": "Requests", "type": "quantitative"},
                "color": {"field": "Category", "type": "nominal"},
                "tooltip": [
                    {"field": "Category", "type": "nominal"},
                    {"field": "Requests", "type": "quantitative"},
                ],
            },
        },
        use_container_width=True,
    )
with right:
    st.subheader("By compound")
    st.bar_chart(filtered["compound"].value_counts().rename("Requests"))
    st.subheader("By status")
    status_counts = filtered["status"].value_counts().rename_axis("Status").reset_index(name="Requests")
    st.vega_lite_chart(
        status_counts,
        {
            "mark": {"type": "arc", "innerRadius": 45},
            "encoding": {
                "theta": {"field": "Requests", "type": "quantitative"},
                "color": {"field": "Status", "type": "nominal"},
                "tooltip": [
                    {"field": "Status", "type": "nominal"},
                    {"field": "Requests", "type": "quantitative"},
                ],
            },
        },
        use_container_width=True,
    )

st.subheader("Request details")
columns = ["ticket_number", "created_at", "compound", "building", "unit_number", "category", "urgency", "status", "issue_description"]
st.dataframe(filtered[columns], use_container_width=True, hide_index=True)
st.download_button(
    "Download filtered requests as CSV",
    filtered[columns].to_csv(index=False).encode("utf-8-sig"),
    file_name="service_requests.csv",
    mime="text/csv",
)
