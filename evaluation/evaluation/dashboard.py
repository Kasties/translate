"""Results dashboard: the result records of every run as bar charts, and the minimal pairs two checkpoints
disagree on.

uv run python -m streamlit run evaluation/dashboard.py

It downloads every run's files from the results repo when a browser tab opens it, then reads results/runs.
"""

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from evaluation.results import (
    MATCH_KEYS, PAIR_SCORES, REPORTED_MINIMAL_PAIR_METRIC, RESULT_RECORDS, download_runs, read_runs,
)

RESULTS_DIR = Path("results")
# One fixed colour per training data, so filtering never repaints one. The colours are the first
# slots of a colour-blind-safe categorical palette; training data not listed here takes the next slot.
TRAINING_DATA_ORDER = ["base model", "control", "seed-grounded", "translated", "MAGPIE"]
# Each colour has a step for the light and the dark theme, and so has the ink of intervals and labels.
COLORS = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}
INK = {"light": "#0b0b0b", "dark": "#ffffff"}
# The chart surface, for the ring that keeps overlapping points apart.
SURFACE = {"light": "#ffffff", "dark": "#0e1117"}

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
# Runs from before 2026-10-02 call the field condition.
records["training_data"] = records.get("training_data", records.get("condition"))
# A base model has no training data; some run configs say so with "none".
records["training_data"] = records["training_data"].replace("none", None).fillna("base model")
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
# Every run that passes the filters, for the charts over time; the bars below show only the latest.
filtered = shown.copy()
# The number a report uses, as in the results table: preference accuracy (mean) for a minimal-pair
# dataset, and EuroEval's first metric, its main one, for a EuroEval dataset.
reported_metrics = {
    dataset: REPORTED_MINIMAL_PAIR_METRIC if suite == "minimal-pair" else group["metric"].iloc[0]
    for (suite, dataset), group in records.groupby(["suite", "dataset"], sort=False)
}
# A checkpoint's quick runs get bars of their own, apart from its full runs.
shown["label"] = shown["reference"].where(shown["run"] == "full", shown["reference"] + " (quick run)")
# A checkpoint run more than once on a dataset shows its latest run, as in the results table.
shown = shown[shown["timestamp"] == shown.groupby(["label", "dataset"])["timestamp"].transform("max")]
kinds = TRAINING_DATA_ORDER + sorted(set(filtered["training_data"]) - set(TRAINING_DATA_ORDER))
shown["training_data_rank"] = shown["training_data"].map(kinds.index)
shown = shown.sort_values(["model_family", "training_data_rank", "label"])
for bound in ["value", "lower", "upper"]:
    shown[f"{bound} (percent)"] = 100 * shown[bound]

theme = "dark" if st.context.theme.type == "dark" else "light"
colors = dict(zip(kinds, COLORS[theme]))
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
        tooltip=["label", "model_family", "training_data", "value (percent)", "lower (percent)",
                 "upper (percent)", "items", "gpu"],
    )  # fmt: skip
    present = [kind for kind in kinds if kind in set(group["training_data"])]
    color = alt.Color(
        "training_data:N",
        scale=alt.Scale(domain=present, range=[colors[kind] for kind in present]),
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
        ["label", "model_family", "training_data", "suite", "dataset", "subset", "metric",
         "value (percent)", "lower (percent)", "upper (percent)", "items", "run", "gpu", "timestamp"]
    ].rename(columns={"label": "checkpoint"}),  # fmt: skip
    column_config={column: st.column_config.NumberColumn(format="%.2f") for column in
                   ["value (percent)", "lower (percent)", "upper (percent)"]},  # fmt: skip
    hide_index=True,
)

st.header("Over time")
st.write(
    "Every run of the checkpoints above, by the time it started, on the number a report uses. A line joins "
    "the checkpoints with the same training data, so it shows how that kind of data does as experiments go on."
)
history = filtered[filtered["metric"] == filtered["dataset"].map(reported_metrics)].copy()
history["run started"] = pd.to_datetime(history["timestamp"])
history["value (percent)"] = 100 * history["value"]
history["label"] = history["reference"].where(history["run"] == "full", history["reference"] + " (quick run)")
history_charts = st.columns(2)
for index, ((dataset, subset, metric), group) in enumerate(
    history.groupby(["dataset", "subset", "metric"], dropna=False)
):
    title = f"{dataset} {metric}" if pd.isna(subset) else f"{dataset} ({subset}) {metric}"
    present = [kind for kind in kinds if kind in set(group["training_data"])]
    color = alt.Color(
        "training_data:N",
        scale=alt.Scale(domain=present, range=[colors[kind] for kind in present]),
        legend=alt.Legend(orient="bottom", title=None),
    )
    points = alt.Chart(group, title=title).encode(
        # Padding in pixels, so the first and last points are not cut by the plot edge.
        x=alt.X("run started:T", title=None, scale=alt.Scale(padding=16)),
        y=alt.Y("value (percent):Q", title="percent", scale=alt.Scale(zero=False)),
        color=color,
    )
    chart = points.mark_line(strokeWidth=2) + points.mark_point(
        # A ring in the surface colour keeps overlapping points apart.
        filled=True, size=90, opacity=1, stroke=SURFACE[theme], strokeWidth=2
    ).encode(
        tooltip=["label", "model_family", "training_data", alt.Tooltip("value (percent):Q", format=".1f"),
                 "run", "gpu", alt.Tooltip("run started:T", format="%Y-%m-%d %H:%M")],
    )  # fmt: skip
    chart = (
        chart.properties(height=260)
        .configure_axis(labelFontSize=14, titleFontSize=14, gridOpacity=0.3)
        .configure_legend(labelFontSize=14)
        .configure_title(fontSize=16)
    )
    history_charts[index % 2].altair_chart(chart, width="stretch")
st.caption("Hover a point for its checkpoint and run.")

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
