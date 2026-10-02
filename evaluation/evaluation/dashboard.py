"""Results dashboard: the result records of every run as bar charts, and the minimal pairs two checkpoints
disagree on.

uv run python -m streamlit run evaluation/dashboard.py

It downloads every run's files from the results repo when a browser tab opens it, then reads results/runs.
"""

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from evaluation.results import MATCH_KEYS, PAIR_SCORES, RESULT_RECORDS, download_runs, read_runs

RESULTS_DIR = Path("results")
# One fixed colour per condition, so filtering never repaints a condition. The colours are the first
# slots of a colour-blind-safe categorical palette; a condition not listed here takes the next slot.
CONDITION_ORDER = ["base model", "control", "seed-grounded", "translated", "MAGPIE"]
# Each colour has a step for the light and the dark theme, and so has the ink of intervals and labels.
COLORS = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}
INK = {"light": "#0b0b0b", "dark": "#ffffff"}

st.set_page_config(page_title="SwenT results", layout="wide")
st.title("SwenT results")

if "downloaded" not in st.session_state:
    try:
        download_runs(RESULTS_DIR)
    except Exception as error:
        st.warning(f"Could not download the runs from the results repo, so only this machine's runs are shown: {error}")
    st.session_state.downloaded = True

# ponytail: reads every run's files on each click; cache them when that gets slow.
records = pd.DataFrame(read_runs(RESULTS_DIR, RESULT_RECORDS))
if records.empty:
    st.error(f"No result records in {RESULTS_DIR / 'runs'}.")
    st.stop()

# A checkpoint without checkpoint metadata is a base model, and its own model family.
records["model_family"] = records["model_family"].fillna(records["checkpoint"])
records["condition"] = records["condition"].fillna("base model")
records["reference"] = records["checkpoint"].where(
    records["revision"].isna(), records["checkpoint"] + "@" + records["revision"].fillna("")
)

filters = st.columns(4)
model_families = filters[0].multiselect(
    "Model family", sorted(records["model_family"].unique()), placeholder="All", key="model_family"
)
suites = filters[1].multiselect("Suite", sorted(records["suite"].unique()), placeholder="All", key="suite")
datasets = filters[2].multiselect("Dataset", sorted(records["dataset"].unique()), placeholder="All", key="dataset")
include_quick_runs = filters[3].checkbox("Include quick runs", key="include_quick_runs")

# An empty filter shows everything.
shown = records
for column, values in [("model_family", model_families), ("suite", suites), ("dataset", datasets)]:
    if values:
        shown = shown[shown[column].isin(values)]
if not include_quick_runs:
    shown = shown[shown["run"] == "full"]
if shown.empty:
    st.info("No result records match the filters. Quick runs are left out unless you tick Include quick runs.")
    st.stop()
shown = shown.copy()
# A checkpoint's quick runs get bars of their own, apart from its full runs.
shown["label"] = shown["reference"].where(shown["run"] == "full", shown["reference"] + " (quick run)")
# A checkpoint run more than once on a dataset shows its latest run, as in the results table.
shown = shown[shown["timestamp"] == shown.groupby(["label", "dataset"])["timestamp"].transform("max")]
conditions = CONDITION_ORDER + sorted(set(shown["condition"]) - set(CONDITION_ORDER))
shown["condition_rank"] = shown["condition"].map(conditions.index)
shown = shown.sort_values(["model_family", "condition_rank", "label"])
for bound in ["value", "lower", "upper"]:
    shown[f"{bound} (percent)"] = 100 * shown[bound]

theme = "dark" if st.context.theme.type == "dark" else "light"
condition_colors = dict(zip(conditions, COLORS[theme]))
# Every value is a fraction shown in percent, so one scale fits every chart; MCC can go below zero.
x_scale = alt.Scale(domain=[min(0, shown["lower (percent)"].min(), shown["value (percent)"].min()), 100])
charts = st.columns(2)
groups = shown.groupby(["dataset", "subset", "metric"], dropna=False)
for index, ((dataset, subset, metric), group) in enumerate(groups):
    # A EuroEval dataset has no subsets.
    title = f"{dataset} {metric}" if pd.isna(subset) else f"{dataset} ({subset}) {metric}"
    # Horizontal bars, so long checkpoint references stay readable.
    bars = alt.Chart(group, title=title).encode(
        y=alt.Y("label:N", sort=list(group["label"]), title=None, axis=alt.Axis(labelLimit=320)),
        tooltip=["label", "model_family", "condition", "value (percent)", "lower (percent)",
                 "upper (percent)", "items", "gpu"],
    )  # fmt: skip
    present = [condition for condition in conditions if condition in set(group["condition"])]
    color = alt.Color(
        "condition:N",
        scale=alt.Scale(domain=present, range=[condition_colors[condition] for condition in present]),
        legend=alt.Legend(orient="bottom", title=None),
    )
    chart = (
        bars.mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
            x=alt.X("value (percent):Q", title="percent", scale=x_scale), color=color
        )
        + bars.mark_errorbar(ticks=True, color=INK[theme], thickness=2).encode(
            x=alt.X("lower (percent):Q", title="percent"), x2="upper (percent):Q"
        )
        # The value in percent, right of the interval, or of the bar when there is no interval.
        + bars.transform_calculate(end="isValid(datum['upper (percent)']) ? datum['upper (percent)'] : datum['value (percent)']")
        .mark_text(align="left", dx=6, color=INK[theme], fontSize=14)
        .encode(x="end:Q", text=alt.Text("value (percent):Q", format=".1f"))
    )
    # A fixed height per bar, so a chart with one bar is as readable as one with ten.
    chart = (
        chart.properties(height=alt.Step(44))
        .configure_axis(labelFontSize=14, titleFontSize=14)
        .configure_legend(labelFontSize=14)
        .configure_title(fontSize=16)
    )
    charts[index % 2].altair_chart(chart, width="stretch")

st.caption("The error bar on each bar is its 95% confidence interval, when the result record has one.")
st.dataframe(
    shown[
        ["label", "model_family", "condition", "suite", "dataset", "subset", "metric",
         "value (percent)", "lower (percent)", "upper (percent)", "items", "run", "gpu", "timestamp"]
    ].rename(columns={"label": "checkpoint"}),  # fmt: skip
    column_config={column: st.column_config.NumberColumn(format="%.2f") for column in
                   ["value (percent)", "lower (percent)", "upper (percent)"]},  # fmt: skip
    hide_index=True,
)

st.header("Minimal pairs two checkpoints disagree on")
pair_scores = pd.DataFrame(read_runs(RESULTS_DIR, PAIR_SCORES))
# The pair scores of the runs behind the charts: one run per checkpoint and dataset.
runs = shown[shown["suite"] == "minimal-pair"][MATCH_KEYS + ["label"]].drop_duplicates()
pairs = pair_scores.merge(runs, on=MATCH_KEYS) if not pair_scores.empty else runs.iloc[:0]
labels = list(pairs["label"].unique())
if len(labels) < 2:
    st.info("Pick filters that show at least two checkpoints with minimal-pair suite results.")
    st.stop()

choices = st.columns(2)
first = choices[0].selectbox("First checkpoint", labels, index=0, key="first_checkpoint")
second = choices[1].selectbox("Second checkpoint", labels, index=1, key="second_checkpoint")
# Preference accuracy (mean): the pair is right when the better sentence has the higher mean
# log-probability per token.
pairs["right"] = (
    pairs["better_summed_log_probability"] / pairs["better_token_count"]
    > pairs["worse_summed_log_probability"] / pairs["worse_token_count"]
)
# pandas cannot merge on lists, so the subsets are joined into one string.
pairs["subsets"] = pairs["subsets"].str.join(", ")
pair_columns = ["dataset", "subsets", "better", "worse"]
both = (
    pairs[pairs["label"] == first]
    .drop_duplicates(pair_columns)
    .merge(pairs[pairs["label"] == second].drop_duplicates(pair_columns), on=pair_columns, suffixes=("_first", "_second"))
)
disagree = both[both["right_first"] != both["right_second"]]
st.write(
    f"{len(disagree)} of {len(both)} pairs, by preference accuracy (mean): only the first checkpoint is right "
    f"on {disagree['right_first'].sum()}, only the second on {disagree['right_second'].sum()}."
)
st.dataframe(
    disagree[pair_columns + ["right_first", "right_second"]].rename(
        columns={
            "better": "better sentence",
            "worse": "worse sentence",
            "right_first": f"{first} prefers the better sentence",
            "right_second": f"{second} prefers the better sentence",
        }
    ),
    hide_index=True,
)
