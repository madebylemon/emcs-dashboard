import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import pointbiserialr

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EMCS Item Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #ffffff; color: #111111; }
    [data-testid="stSidebar"] { background-color: #f5f5f5; border-right: 1px solid #ddd; }
    [data-testid="stSidebar"] * { color: #111111 !important; }

    .kpi-row { display: flex; gap: 12px; margin-bottom: 24px; flex-wrap: wrap; }
    .kpi-card {
        flex: 1; min-width: 150px;
        background: #f9f9f9; border: 1px solid #ddd; border-radius: 6px;
        padding: 16px 12px; text-align: center;
    }
    .kpi-label { font-size: 11px; color: #555; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
    .kpi-value { font-size: 28px; font-weight: 700; color: #111; }
    .kpi-value.warn { color: #111; border-bottom: 2px solid #111; display: inline-block; }

    .stTabs [data-baseweb="tab-list"] { border-bottom: 2px solid #ddd; gap: 0; }
    .stTabs [data-baseweb="tab"] { background: transparent; color: #666; font-weight: 500; border-radius: 0; }
    .stTabs [aria-selected="true"] { color: #111 !important; border-bottom: 2px solid #111; }

    .flag-card {
        background: #f9f9f9; border-left: 3px solid #111;
        padding: 14px 18px; margin-bottom: 12px; border-radius: 0 4px 4px 0;
    }
    .flag-card h4 { color: #111; margin: 0 0 6px 0; font-size: 14px; }
    .flag-card p  { color: #333; margin: 0; font-size: 13px; line-height: 1.6; }

    .main-header {
        border-bottom: 2px solid #111; padding: 16px 0 12px 0; margin-bottom: 20px;
    }
    .main-header h1 { color: #111; margin: 0; font-size: 24px; font-weight: 700; }
    .main-header p  { color: #555; margin: 4px 0 0 0; font-size: 13px; }
    .stMarkdown hr  { border-color: #ddd; }

    .stApp * { color: #111111; }
    .stApp p, .stApp span, .stApp div, .stApp label,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 { color: #111111 !important; }
    [data-testid="column"] * { color: #111111 !important; }
    .stMarkdown p, .stMarkdown span, .stMarkdown div { color: #111111 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
TYPE_COLORS = {
    "E":   "#2196F3",   # blue   — Energy
    "M":   "#F44336",   # red    — Momentum
    "E&M": "#9C27B0",   # purple — Both
}
TYPE_COLORS_LIGHT = {
    "E":   "rgba(33,150,243,0.35)",
    "M":   "rgba(244,67,54,0.35)",
    "E&M": "rgba(156,39,176,0.35)",
}
TYPE_LABELS  = {"E": "Energy", "M": "Momentum", "E&M": "Energy & Momentum"}
PLOT_BG      = "#ffffff"
PAPER_BG     = "#ffffff"
GRID_CLR     = "#eeeeee"
AXIS_CLR     = "#333333"
FLAG_CLR     = "#111111"
GAIN_POS_CLR = "#27ae60"   # green  — positive gain
GAIN_NEG_CLR = "#e74c3c"   # red    — negative gain

# ─── Data ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("emcs_data.csv")
    df["type_label"] = df["type"].map(TYPE_LABELS)
    # Compute 'problematic' from thresholds so it's never missing even if CSV lacks the column
    df["problematic"] = (
        (df["ctt_diff"]         < 0.20) |
        (df["ctt_disc"]         < 0.20) |
        (df["point_biserial"]   < 0.20) |
        (~df["irt_disc"].between(0.50, 2.50)) |
        (df["irt_guess"]        > 0.25) |
        (df.get("alpha_if_removed", pd.Series(0, index=df.index)) > 0.7563)
    )
    return df

df_full = load_data()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Controls")
    st.markdown("---")

    selected_types = st.multiselect(
        "Filter by Item Type",
        options=list(TYPE_LABELS.values()),
        default=list(TYPE_LABELS.values()),
    )
    show_problematic = st.checkbox("Highlight Flagged Items", value=True)
    show_thresholds  = st.checkbox("Show Threshold Lines", value=True)

    st.markdown("---")
    st.markdown("### Psychometric Thresholds")
    st.markdown("""\
| Metric | Threshold |
|---|---|
| CTT Difficulty | ≥ 0.20 |
| CTT Discrimination | ≥ 0.20 |
| Point-Biserial | ≥ 0.20 |
| IRT Discrimination | 0.50 – 2.50 |
| IRT Guessing | ≤ 0.25 |
""")
    alpha_threshold = st.number_input(
        "Alpha if Removed ≤",
        min_value=0.0, max_value=1.0,
        value=0.7563, step=0.0001, format="%.4f",
    )

    st.markdown("---")
    with st.expander("📖 Metric Glossary"):
        st.markdown("""
**Avg Pre-Test**  
Mean proportion correct across all items *before* instruction.  
Computed as the average of each item's pre-test p-value (% students correct).

**Avg Post-Test**  
Mean proportion correct across all items *after* instruction.

**Avg Normalized Gain (g)**  
Measures learning efficiency relative to potential improvement:
> *g = (post − pre) / (1 − pre)*  

- g ≈ 0.7 → high gain · g ≈ 0.3 → medium · g < 0 → students scored worse

**CTT Difficulty (p-value)**  
Proportion of students who answered correctly. Higher = easier item.

**CTT Discrimination**  
Correlation between item score and total test score. Higher = item better separates high/low performers.

**Point-Biserial**  
Pearson correlation of a binary item score with the continuous total score.

**IRT Parameters (3PL model)**  
- *a* — Discrimination: steepness of the ICC curve  
- *b* — Difficulty: θ at which P(correct) = 0.5  
- *c* — Guessing: lower asymptote (chance-level probability)

**Alpha if Removed**  
Cronbach's α of the scale if this item were deleted. Values above the overall α indicate the item reduces reliability.

**Color coding:** 🔵 Energy &nbsp; 🔴 Momentum &nbsp; 🟣 Energy & Momentum
""")

# ─── Filter + Editable Session State ────────────────────────────────────────
_filter_key = tuple(sorted(selected_types))
_base_df = df_full[df_full["type_label"].isin(selected_types)].copy()

# Reset editable data when the type filter changes
if ("edited_df" not in st.session_state
        or st.session_state.get("filter_key") != _filter_key):
    st.session_state.edited_df = _base_df.copy()
    st.session_state.filter_key = _filter_key

# All charts use the editable df so edits propagate automatically
df = st.session_state.edited_df

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>EMCS Item Explorer</h1>
  <p>Psychometric Dashboard — Energy &amp; Momentum Conceptual Survey</p>
</div>
""", unsafe_allow_html=True)

# ─── KPI Cards ────────────────────────────────────────────────────────────────
avg_pre    = df["pre_test"].mean()
avg_post   = df["post_test"].mean()
avg_gain   = df["gain"].mean()
n_prob     = int(df["problematic"].sum())
n_neg_gain = int((df["gain"] < 0).sum())

st.markdown(f"""
<div class="kpi-row">
  <div class="kpi-card">
    <div class="kpi-label">Avg Pre-Test</div>
    <div class="kpi-value">{avg_pre:.2f}</div>
    <div class="kpi-label" style="font-size:10px;margin-top:4px">mean item p-value before instruction</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Avg Post-Test</div>
    <div class="kpi-value">{avg_post:.2f}</div>
    <div class="kpi-label" style="font-size:10px;margin-top:4px">mean item p-value after instruction</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Avg Norm. Gain</div>
    <div class="kpi-value">{avg_gain:.2f}</div>
    <div class="kpi-label" style="font-size:10px;margin-top:4px">g = (post−pre)/(1−pre)</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Problematic Items</div>
    <div class="kpi-value {'warn' if n_prob > 0 else ''}">{n_prob}</div>
    <div class="kpi-label" style="font-size:10px;margin-top:4px">items breaching ≥1 threshold</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Negative Gain Items</div>
    <div class="kpi-value {'warn' if n_neg_gain > 0 else ''}">{n_neg_gain}</div>
    <div class="kpi-label" style="font-size:10px;margin-top:4px">post-test score below pre-test</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Shared layout defaults ───────────────────────────────────────────────────
LAYOUT_BASE = dict(
    paper_bgcolor=PAPER_BG,
    plot_bgcolor=PLOT_BG,
    font=dict(color=AXIS_CLR, size=12),
    margin=dict(l=60, r=40, t=40, b=60),
)

def clean_axes(fig, rows=1):
    for i in range(1, rows + 1):
        suffix = "" if i == 1 else str(i)
        for ax in [f"xaxis{suffix}", f"yaxis{suffix}"]:
            if ax in fig.layout:
                fig.layout[ax].update(
                    showgrid=True, gridcolor=GRID_CLR, gridwidth=1,
                    linecolor="#aaa", linewidth=1,
                    tickcolor="#aaa", tickfont=dict(color=AXIS_CLR),
                    title_font=dict(color=AXIS_CLR),
                    zeroline=False,
                )

LEGEND_STYLE = dict(bgcolor="#fff", bordercolor="#ddd", borderwidth=1, font=dict(color="#111111"))

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Tab 1 — Discrimination vs Difficulty",
    "Tab 2 — Pre/Post & Gain",
    "Tab 3 — IRT Scatter",
    "Tab 4 — Full Metrics Table",
    "Tab 5 — Item Characteristic Curves",
    "Tab 6 — Item Category Analysis",
    "Tab 7 — Compute from Raw Data",
])

# ════════════════════════════════════════════
# TAB 1 — CTT Scatter (uniform dot size)
# ════════════════════════════════════════════
with tab1:
    st.markdown("### CTT Discrimination vs Difficulty")
    st.caption("Shaded region = acceptable zone (both > 0.20) · Color = item type")

    fig1 = go.Figure()

    if show_thresholds:
        fig1.add_shape(
            type="rect", x0=0.20, x1=1.0, y0=0.20, y1=1.0,
            fillcolor="rgba(0,0,0,0.04)", line=dict(color="rgba(0,0,0,0.25)", width=1),
            layer="below",
        )
        fig1.add_annotation(
            x=0.60, y=0.95, text="Acceptable Zone",
            showarrow=False, font=dict(color="#555", size=10),
        )
        fig1.add_vline(x=0.20, line=dict(color="#333", width=1, dash="dash"))
        fig1.add_hline(y=0.20, line=dict(color="#333", width=1, dash="dash"))

    for t_key, t_label in TYPE_LABELS.items():
        sub = df[df["type"] == t_key]
        if sub.empty:
            continue
        is_prob = sub["problematic"] if show_problematic else pd.Series([False]*len(sub), index=sub.index)
        fig1.add_trace(go.Scatter(
            x=sub["ctt_diff"], y=sub["ctt_disc"],
            mode="markers",
            name=t_label,
            marker=dict(
                size=12,
                color=TYPE_COLORS[t_key],
                opacity=0.85,
                line=dict(
                    color=[FLAG_CLR if p else TYPE_COLORS[t_key] for p in is_prob],
                    width=[3 if p else 1 for p in is_prob],
                ),
            ),
            text=sub["item"],
            customdata=sub[["type_label"]].values,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "CTT Difficulty: %{x:.3f}<br>"
                "CTT Discrimination: %{y:.3f}<br>"
                "Type: %{customdata[0]}<extra></extra>"
            ),
        ))

    if show_problematic:
        for _, row in df[df["problematic"]].iterrows():
            fig1.add_annotation(
                x=row["ctt_diff"], y=row["ctt_disc"],
                text=f"  {row['item']}",
                showarrow=True, arrowhead=2, arrowcolor=FLAG_CLR,
                font=dict(color=FLAG_CLR, size=11, family="monospace"),
                ax=40, ay=-30,
            )

    fig1.update_layout(
        **LAYOUT_BASE,
        legend=LEGEND_STYLE,
        xaxis=dict(title="CTT Difficulty (p-value)", range=[-0.02, 1.02]),
        yaxis=dict(title="CTT Discrimination", range=[-0.05, 0.70]),
        height=520,
    )
    clean_axes(fig1)
    st.plotly_chart(fig1, use_container_width=True)

# ════════════════════════════════════════════
# TAB 2 — Pre/Post + Gain (green/red diamonds)
# ════════════════════════════════════════════
with tab2:
    st.markdown("### Pre-Test vs Post-Test Correctness with Normalized Gain")
    st.caption("Light bar = Pre-Test · Dark bar = Post-Test · Diamond = Normalized Gain · 🟢 Positive / 🔴 Negative")

    fig2 = make_subplots(specs=[[{"secondary_y": True}]])

    pre_clrs  = [TYPE_COLORS_LIGHT[t] for t in df["type"]]
    post_clrs = [TYPE_COLORS[t]       for t in df["type"]]

    fig2.add_trace(go.Bar(
        name="Pre-Test", x=df["item"], y=df["pre_test"],
        marker_color=pre_clrs, opacity=0.9,
        hovertemplate="<b>%{x}</b><br>Pre-Test: %{y:.2f}<extra></extra>",
    ), secondary_y=False)

    fig2.add_trace(go.Bar(
        name="Post-Test", x=df["item"], y=df["post_test"],
        marker_color=post_clrs, opacity=0.9,
        hovertemplate="<b>%{x}</b><br>Post-Test: %{y:.2f}<extra></extra>",
    ), secondary_y=False)

    # Gain markers — green if positive, red if negative
    for _, row in df.iterrows():
        clr = GAIN_POS_CLR if row["gain"] >= 0 else GAIN_NEG_CLR
        fig2.add_trace(go.Scatter(
            x=[row["item"]], y=[row["gain"]],
            mode="markers",
            showlegend=False,
            marker=dict(
                color=clr, size=10, symbol="diamond",
                line=dict(color="#fff", width=1),
            ),
            hovertemplate=f"<b>{row['item']}</b><br>Gain: {row['gain']:.2f}<extra></extra>",
        ), secondary_y=True)

    # Legend entries for gain
    for clr, label in [(GAIN_POS_CLR, "Gain ≥ 0"), (GAIN_NEG_CLR, "Gain < 0")]:
        fig2.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name=label,
            marker=dict(color=clr, size=10, symbol="diamond"),
            showlegend=True,
        ), secondary_y=True)

    # Type legend entries
    for t_key, t_label in TYPE_LABELS.items():
        fig2.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name=t_label,
            marker=dict(color=TYPE_COLORS[t_key], size=10, symbol="square"),
            showlegend=True,
        ), secondary_y=False)

    if show_thresholds:
        fig2.add_hline(y=0, line=dict(color="#888", width=1, dash="dot"), secondary_y=True)

    fig2.update_layout(
        **LAYOUT_BASE,
        barmode="group",
        legend=LEGEND_STYLE,
        height=510,
        xaxis=dict(tickangle=-45),
    )
    fig2.update_yaxes(title_text="Proportion Correct", secondary_y=False, range=[0, 1.05],
                      showgrid=True, gridcolor=GRID_CLR, linecolor="#aaa")
    fig2.update_yaxes(title_text="Normalized Gain", secondary_y=True, range=[-0.5, 0.7],
                      showgrid=False, linecolor="#aaa")
    st.plotly_chart(fig2, use_container_width=True)

# ════════════════════════════════════════════
# TAB 3 — IRT Discrimination vs Difficulty scatter
# ════════════════════════════════════════════
with tab3:
    st.markdown("### IRT Discrimination vs. Difficulty")
    st.caption("Dot size = IRT guessing (c) · Color = item type · Dashed lines = acceptable range")

    fig3 = go.Figure()

    if show_thresholds:
        # Acceptable discrimination band (0.50 – 2.50)
        fig3.add_hrect(y0=0.50, y1=2.50,
                       fillcolor="rgba(0,0,0,0.04)",
                       line=dict(color="rgba(0,0,0,0.25)", width=1),
                       layer="below")
        fig3.add_annotation(x=0.5, y=1.50, text="Acceptable Discrimination Zone",
                            showarrow=False, font=dict(color="#555", size=10))
        fig3.add_hline(y=0.50, line=dict(color="#333", width=1, dash="dash"))
        fig3.add_hline(y=2.50, line=dict(color="#333", width=1, dash="dash"))

    for t_key, t_label in TYPE_LABELS.items():
        sub = df[df["type"] == t_key]
        if sub.empty:
            continue
        is_prob = sub["problematic"] if show_problematic else pd.Series([False]*len(sub), index=sub.index)
        fig3.add_trace(go.Scatter(
            x=sub["irt_diff"], y=sub["irt_disc"],
            mode="markers",
            name=t_label,
            marker=dict(
                size=sub["irt_guess"] * 80 + 8,
                color=TYPE_COLORS[t_key],
                opacity=0.82,
                line=dict(
                    color=[FLAG_CLR if p else TYPE_COLORS[t_key] for p in is_prob],
                    width=[3 if p else 1 for p in is_prob],
                ),
            ),
            text=sub["item"],
            customdata=sub[["irt_guess", "type_label"]].values,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "IRT Difficulty (b): %{x:.3f}<br>"
                "IRT Discrimination (a): %{y:.3f}<br>"
                "IRT Guessing (c): %{customdata[0]:.3f}<br>"
                "Type: %{customdata[1]}<extra></extra>"
            ),
        ))

    if show_problematic:
        for _, row in df[df["problematic"]].iterrows():
            fig3.add_annotation(
                x=row["irt_diff"], y=row["irt_disc"],
                text=f"  {row['item']}",
                showarrow=True, arrowhead=2, arrowcolor=FLAG_CLR,
                font=dict(color=FLAG_CLR, size=11, family="monospace"),
                ax=40, ay=-30,
            )

    fig3.update_layout(
        **LAYOUT_BASE,
        legend=LEGEND_STYLE,
        xaxis=dict(title="IRT Difficulty (b)"),
        yaxis=dict(title="IRT Discrimination (a)", range=[-0.1, 3.2]),
        height=520,
    )
    clean_axes(fig3)
    st.plotly_chart(fig3, use_container_width=True)

# ════════════════════════════════════════════
# TAB 4 — Full Table + Download + Flagged Cards
# ════════════════════════════════════════════
with tab4:
    st.markdown("### Full Psychometric Metrics")

    # ── CSV Upload: auto-loads into all charts ──────────────────────────────
    with st.expander("\u2b06\ufe0f Upload CSV to replace data (all charts update automatically)"):
        uploaded_csv = st.file_uploader(
            "Upload a metrics CSV (must have columns: item, type, pre_test, post_test, gain, "
            "ctt_diff, ctt_disc, point_biserial, irt_diff, irt_disc, irt_guess, alpha_if_removed)",
            type=["csv"], key="tab4_csv_upload",
        )
        if uploaded_csv is not None:
            # Unique ID for this specific file — prevents reprocessing on every rerun
            # (st.file_uploader keeps the file selected across reruns, which would
            #  cause an infinite rerun loop if we called st.rerun() unconditionally)
            _file_id = f"{uploaded_csv.name}_{uploaded_csv.size}"

            if st.session_state.get("tab4_last_csv_id") != _file_id:
                # First time seeing this file: parse and load it
                try:
                    new_df = pd.read_csv(uploaded_csv)
                    required_cols = ["item", "type", "pre_test", "post_test", "gain",
                                      "ctt_diff", "ctt_disc", "point_biserial",
                                      "irt_diff", "irt_disc", "irt_guess", "alpha_if_removed"]
                    missing = [c for c in required_cols if c not in new_df.columns]
                    if missing:
                        st.error(f"\u274c CSV is missing columns: {missing}")
                    else:
                        new_df["type_label"] = new_df["type"].map(TYPE_LABELS)
                        new_df["problematic"] = (
                            (new_df["ctt_diff"]         < 0.20) |
                            (new_df["ctt_disc"]         < 0.20) |
                            (new_df["point_biserial"]   < 0.20) |
                            (~new_df["irt_disc"].between(0.50, 2.50)) |
                            (new_df["irt_guess"]        > 0.25) |
                            (new_df["alpha_if_removed"] > alpha_threshold)
                        )
                        st.session_state.edited_df        = new_df.reset_index(drop=True)
                        # Use the CURRENT filter key so the top-of-script guard does NOT
                        # reset edited_df back to the baseline on the next rerun
                        st.session_state.filter_key       = _filter_key
                        st.session_state.tab4_last_csv_id = _file_id
                        # One clean rerun to redraw all charts with the new data
                        st.rerun()
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")
            else:
                # File already processed — show stable confirmation, no rerun needed
                st.success(
                    f"\u2705 **{uploaded_csv.name}** is loaded \u2014 "
                    f"{len(st.session_state.edited_df)} items active. "
                    "All charts reflect this data."
                )



    st.caption("Click any numeric cell to edit it — all charts update automatically on change")

    edit_cols = [
        "item", "type", "pre_test", "post_test", "gain",
        "ctt_diff", "ctt_disc", "point_biserial",
        "irt_diff", "irt_disc", "irt_guess", "alpha_if_removed",
    ]

    edited = st.data_editor(
        st.session_state.edited_df[edit_cols].reset_index(drop=True),
        use_container_width=True,
        height=500,
        num_rows="fixed",
        disabled=["item", "type"],
        column_config={
            "item":             st.column_config.TextColumn("Item", disabled=True),
            "type":             st.column_config.TextColumn("Type", disabled=True),
            "pre_test":         st.column_config.NumberColumn("Pre-Test",            format="%.3f", min_value=0.0, max_value=1.0),
            "post_test":        st.column_config.NumberColumn("Post-Test",           format="%.3f", min_value=0.0, max_value=1.0),
            "gain":             st.column_config.NumberColumn("Norm. Gain",          format="%.3f", min_value=-1.0, max_value=1.0),
            "ctt_diff":         st.column_config.NumberColumn("CTT Difficulty",      format="%.3f", min_value=0.0, max_value=1.0),
            "ctt_disc":         st.column_config.NumberColumn("CTT Discrimination",  format="%.3f"),
            "point_biserial":   st.column_config.NumberColumn("Point-Biserial",      format="%.3f"),
            "irt_diff":         st.column_config.NumberColumn("IRT Difficulty (b)",  format="%.3f"),
            "irt_disc":         st.column_config.NumberColumn("IRT Discrimination (a)", format="%.3f", min_value=0.0),
            "irt_guess":        st.column_config.NumberColumn("IRT Guessing (c)",    format="%.3f", min_value=0.0, max_value=1.0),
            "alpha_if_removed": st.column_config.NumberColumn("Alpha if Removed",    format="%.4f"),
        },
        key="metrics_editor",
    )

    # ── Sync edits back to session state ─────────────────────────────────────
    numeric_cols = [c for c in edit_cols if c not in ("item", "type")]
    for col in numeric_cols:
        st.session_state.edited_df[col] = edited[col].values

    # Recompute 'problematic' flag from current thresholds
    s = st.session_state.edited_df
    s["problematic"] = (
        (s["ctt_diff"]       < 0.20) |
        (s["ctt_disc"]       < 0.20) |
        (s["point_biserial"] < 0.20) |
        (~s["irt_disc"].between(0.50, 2.50)) |
        (s["irt_guess"]      > 0.25) |
        (s["alpha_if_removed"] > alpha_threshold)
    )

    # ── Buttons row ───────────────────────────────────────────────────────────
    col_dl, col_reset = st.columns([3, 1])
    with col_dl:
        csv_bytes = edited.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇ Download current table as CSV",
            data=csv_bytes,
            file_name="emcs_metrics.csv",
            mime="text/csv",
        )
    with col_reset:
        if st.button("↺ Reset to original data"):
            st.session_state.edited_df = _base_df.copy()
            st.rerun()

    st.markdown("---")
    st.markdown("### Flagged Item Analysis")

    st.markdown("""
<div class="flag-card">
  <h4>Q16 — Low Discrimination &amp; High Guessing</h4>
  <p>
    Q16 shows a <strong>CTT discrimination of 0.15</strong> (threshold: 0.20) and a
    <strong>point-biserial of 0.11</strong>, indicating poor differentiation between
    high- and low-ability students. The <strong>IRT a-parameter (0.32)</strong> falls below
    the minimum of 0.50, and the guessing parameter <strong>(c = 0.32)</strong> exceeds the
    0.25 ceiling. <strong>Alpha-if-removed (0.7612) &gt; overall alpha (0.7563)</strong>,
    so removing this item would improve scale reliability.
  </p>
</div>
<div class="flag-card">
  <h4>Q22 — Extremely Low Difficulty &amp; Discrimination</h4>
  <p>
    Q22 has a <strong>CTT difficulty of 0.22</strong> and a
    <strong>discrimination index of 0.08</strong>—well below the 0.20 threshold.
    The <strong>IRT b-parameter of −1.12</strong> marks it as extremely easy, while the
    <strong>a-parameter (0.18)</strong> indicates near-zero discriminatory power.
    The <strong>point-biserial (0.07)</strong> shows no meaningful correlation with total score,
    and <strong>alpha-if-removed (0.7598)</strong> confirms it degrades reliability.
  </p>
</div>
<div class="flag-card">
  <h4>Q23 — Negative Gain &amp; Out-of-Range IRT Discrimination</h4>
  <p>
    Q23 is the only item with a <strong>negative normalized gain (−0.13)</strong>—students scored
    worse on the post-test than the pre-test. The <strong>IRT a-parameter (2.84)</strong> exceeds
    the 2.50 upper bound, suggesting over-discrimination or a keying error. The
    <strong>guessing parameter (c = 0.28)</strong> also exceeds 0.25, and
    <strong>alpha-if-removed (0.7578 &gt; 0.7563)</strong> indicates a negative
    contribution to scale reliability.
  </p>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════
# TAB 5 — Item Characteristic Curves (3PL)
# ════════════════════════════════════════════
with tab5:
    st.markdown("### Item Characteristic Curves")
    st.caption(
        "3PL model: P(θ) = c + (1−c) / (1 + exp(−a·(θ−b)))  ·  "
        "Color = item type  ·  Red dashed border = flagged item"
    )

    theta = np.linspace(-3, 3, 200)
    items = df.reset_index(drop=True)

    NCOLS, NROWS = 5, 5
    subplot_titles = [row["item"] for _, row in items.iterrows()]

    fig5 = make_subplots(
        rows=NROWS, cols=NCOLS,
        subplot_titles=subplot_titles,
        shared_xaxes=False,
        shared_yaxes=False,
        horizontal_spacing=0.055,
        vertical_spacing=0.09,
    )

    for idx, (_, row) in enumerate(items.iterrows()):
        r     = idx // NCOLS + 1
        c_col = idx  % NCOLS + 1
        a, b, c = row["irt_disc"], row["irt_diff"], row["irt_guess"]
        prob  = c + (1 - c) / (1 + np.exp(-a * (theta - b)))
        color = TYPE_COLORS[row["type"]]
        ax_idx = "" if idx == 0 else str(idx + 1)

        # ICC curve
        fig5.add_trace(go.Scatter(
            x=theta, y=prob,
            mode="lines",
            line=dict(color=color, width=2),
            showlegend=False,
            hovertemplate=(
                f"<b>{row['item']}</b><br>"
                f"a={a:.2f}, b={b:.2f}, c={c:.2f}<br>"
                "θ: %{x:.2f}<br>P(θ): %{y:.3f}<extra></extra>"
            ),
        ), row=r, col=c_col)

        # a / b / c parameter label inside panel (bottom-right)
        fig5.add_annotation(
            xref=f"x{ax_idx}", yref=f"y{ax_idx}",
            x=2.6, y=0.08,
            text=f"a={a:.2f}<br>b={b:.2f}<br>c={c:.2f}",
            showarrow=False, align="right",
            font=dict(size=7.5, color=color),
            bgcolor="rgba(255,255,255,0.80)",
        )

        # Flagged items: red dashed border
        if row["problematic"] and show_problematic:
            fig5.add_shape(
                type="rect",
                xref=f"x{ax_idx}", yref=f"y{ax_idx}",
                x0=-3, x1=3, y0=0, y1=1,
                line=dict(color="#e74c3c", width=1.5, dash="dash"),
                fillcolor="rgba(0,0,0,0)", layer="above",
            )

        # Dotted reference lines (θ=0 and P=0.5)
        if show_thresholds:
            fig5.add_shape(type="line",
                xref=f"x{ax_idx}", yref=f"y{ax_idx}",
                x0=-3, x1=3, y0=0.5, y1=0.5,
                line=dict(color="#ddd", width=0.8, dash="dot"), layer="below")
            fig5.add_shape(type="line",
                xref=f"x{ax_idx}", yref=f"y{ax_idx}",
                x0=0, x1=0, y0=0, y1=1,
                line=dict(color="#ddd", width=0.8, dash="dot"), layer="below")

    # Apply uniform axis styling to all 25 panels
    for idx in range(NROWS * NCOLS):
        r     = idx // NCOLS + 1
        c_col = idx  % NCOLS + 1
        ax_idx = "" if idx == 0 else str(idx + 1)
        fig5.update_layout(**{
            f"xaxis{ax_idx}": dict(
                range=[-3, 3], showgrid=True, gridcolor=GRID_CLR,
                linecolor="#ccc", tickfont=dict(size=6.5),
                zeroline=False,
                title_text="θ" if r == NROWS else "",
                title_font=dict(size=8, color=AXIS_CLR),
                dtick=1,
            ),
            f"yaxis{ax_idx}": dict(
                range=[0, 1], showgrid=True, gridcolor=GRID_CLR,
                linecolor="#ccc", tickfont=dict(size=6.5),
                zeroline=False,
                title_text="P(θ)" if c_col == 1 else "",
                title_font=dict(size=8, color=AXIS_CLR),
                dtick=0.5,
            ),
        })

    # Subplot title styling (set via annotations produced by make_subplots)
    for ann in fig5.layout.annotations:
        ann.font.size  = 10
        ann.font.color = "#111111"

    fig5.update_layout(
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(color=AXIS_CLR, size=9),
        height=950,
        margin=dict(l=45, r=15, t=55, b=40),
        showlegend=False,
    )

    # Manual color legend below chart
    legend_html = "&nbsp;&nbsp;".join(
        f'<span style="color:{TYPE_COLORS[k]};font-weight:700">■ {v}</span>'
        for k, v in TYPE_LABELS.items()
    )
    legend_html += '&nbsp;&nbsp;&nbsp;<span style="color:#e74c3c;font-weight:700">⬚ Flagged item</span>'
    st.markdown(legend_html, unsafe_allow_html=True)

    st.plotly_chart(fig5, use_container_width=True)

# ════════════════════════════════════════════
# TAB 6 — Item Analysis by Category
# ════════════════════════════════════════════
with tab6:
    st.markdown("### Item Analysis by Category")
    st.caption("Compare item performance across Energy, Momentum, and Energy & Momentum question types")

    # ── Panel A: Grouped avg pre / post / gain by type ───────────────────────
    st.markdown("#### Average Scores by Item Type")

    type_summary = (
        df.groupby("type_label")[["pre_test", "post_test", "gain"]]
        .mean()
        .reset_index()
        .rename(columns={"type_label": "Type", "pre_test": "Pre-Test",
                         "post_test": "Post-Test", "gain": "Norm. Gain"})
    )
    # Preserve display order
    order = [TYPE_LABELS[k] for k in TYPE_LABELS if TYPE_LABELS[k] in type_summary["Type"].values]
    type_summary["Type"] = pd.Categorical(type_summary["Type"], categories=order, ordered=True)
    type_summary = type_summary.sort_values("Type")

    fig6a = go.Figure()
    bar_colors = {TYPE_LABELS[k]: TYPE_COLORS[k] for k in TYPE_LABELS}
    light_colors = {TYPE_LABELS[k]: TYPE_COLORS_LIGHT[k] for k in TYPE_LABELS}

    for _, row in type_summary.iterrows():
        t = row["Type"]
        clr = bar_colors.get(t, "#888")
        lclr = light_colors.get(t, "rgba(136,136,136,0.35)")
        fig6a.add_trace(go.Bar(
            name=f"{t} — Pre",  x=[t], y=[row["Pre-Test"]],
            marker_color=lclr, showlegend=True,
            hovertemplate=f"<b>{t}</b><br>Pre-Test: {row['Pre-Test']:.3f}<extra></extra>",
        ))
        fig6a.add_trace(go.Bar(
            name=f"{t} — Post", x=[t], y=[row["Post-Test"]],
            marker_color=clr, showlegend=True,
            hovertemplate=f"<b>{t}</b><br>Post-Test: {row['Post-Test']:.3f}<extra></extra>",
        ))
        fig6a.add_trace(go.Scatter(
            name=f"{t} — Gain", x=[t], y=[row["Norm. Gain"]],
            mode="markers",
            marker=dict(color=clr, size=14, symbol="diamond",
                        line=dict(color="#fff", width=2)),
            yaxis="y2", showlegend=True,
            hovertemplate=f"<b>{t}</b><br>Norm. Gain: {row['Norm. Gain']:.3f}<extra></extra>",
        ))

    fig6a.update_layout(
        **LAYOUT_BASE,
        barmode="group",
        legend=LEGEND_STYLE,
        height=420,
        xaxis=dict(title="Item Type"),
        yaxis=dict(title="Proportion Correct", range=[0, 1.05],
                   showgrid=True, gridcolor=GRID_CLR),
        yaxis2=dict(title="Normalized Gain", overlaying="y", side="right",
                    range=[-0.2, 0.7], showgrid=False),
    )
    st.plotly_chart(fig6a, use_container_width=True)

    # ── Panel B: Pre vs Post scatter per item, labeled and colored by type ───
    st.markdown("#### Pre-Test vs Post-Test per Item (colored by type)")
    st.caption("Each point = one question · Diagonal = no change line · Above diagonal = learning gain")

    fig6b = go.Figure()

    # No-change diagonal
    fig6b.add_shape(type="line", x0=0, x1=1, y0=0, y1=1,
                    line=dict(color="#ccc", width=1, dash="dot"), layer="below")

    for t_key, t_label in TYPE_LABELS.items():
        sub = df[df["type"] == t_key]
        if sub.empty:
            continue
        fig6b.add_trace(go.Scatter(
            x=sub["pre_test"], y=sub["post_test"],
            mode="markers+text",
            name=t_label,
            text=sub["item"],
            textposition="top center",
            textfont=dict(size=9, color=TYPE_COLORS[t_key]),
            marker=dict(
                size=11, color=TYPE_COLORS[t_key], opacity=0.85,
                line=dict(
                    color=[FLAG_CLR if p else TYPE_COLORS[t_key] for p in sub["problematic"]],
                    width=[3 if p else 1 for p in sub["problematic"]],
                ),
            ),
            customdata=sub[["gain", "type_label"]].values,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Pre-Test: %{x:.3f}<br>"
                "Post-Test: %{y:.3f}<br>"
                "Norm. Gain: %{customdata[0]:.3f}<br>"
                "Type: %{customdata[1]}<extra></extra>"
            ),
        ))

    fig6b.update_layout(
        **LAYOUT_BASE,
        legend=LEGEND_STYLE,
        height=520,
        xaxis=dict(title="Pre-Test (proportion correct)", range=[-0.02, 1.02],
                   showgrid=True, gridcolor=GRID_CLR, linecolor="#aaa"),
        yaxis=dict(title="Post-Test (proportion correct)", range=[-0.02, 1.02],
                   showgrid=True, gridcolor=GRID_CLR, linecolor="#aaa"),
    )
    st.plotly_chart(fig6b, use_container_width=True)

    # ── Note on unavailable analyses ─────────────────────────────────────────
    st.markdown("---")
    with st.expander("ℹ️ Analyses requiring raw student data (not currently available)"):
        st.markdown("""
The following analyses would significantly enhance this dashboard but require **individual student response data** beyond the item-level summary in `emcs_data.csv`:

| Analysis | What's needed |
|---|---|
| **Student-level distributions** (histograms, boxplots) | Per-student scores on each item |
| **Demographic filters** (gender, year in course) | Student metadata linked to responses |
| **Exploratory Factor Analysis (EFA)** | Full inter-item correlation matrix from raw responses |
| **Reliability interval estimates** | Bootstrap resampling from raw data |

If raw anonymized response data is made available, these sections can be activated automatically.
""")

# ════════════════════════════════════════════
# TAB 7 — Compute Metrics from Raw xlsx
# ════════════════════════════════════════════
with tab7:
    st.markdown("### Compute Psychometric Metrics from Raw Response File")
    st.caption(
        "Upload a student response xlsx (columns: file_name, CODE, Q01–Q25). "
        "The system scores responses, computes all CTT / IRT metrics, and lets you "
        "download the result as a CSV to import into Tab 4."
    )

    # ── Answer Key (hardcoded, shown for transparency) ──────────────────────
    ANSWER_KEY_T7 = {
        "Q01":"b","Q02":"e","Q03":"b","Q04":"a","Q05":"d",
        "Q06":"c","Q07":"e","Q08":"c","Q09":"a","Q10":"d",
        "Q11":"e","Q12":"d","Q13":"c","Q14":"d","Q15":"a",
        "Q16":"c","Q17":"b","Q18":"e","Q19":"b","Q20":"a",
        "Q21":"c","Q22":"d","Q23":"b","Q24":"a","Q25":"e",
    }
    with st.expander("🔑 Answer Key (click to view)"):
        ak_df = pd.DataFrame([ANSWER_KEY_T7])
        st.dataframe(ak_df, use_container_width=True, hide_index=True)

    # ── File Upload ────────────────────────────────────────────────────────────
    uploaded_raw = st.file_uploader(
        "Upload student response file (.xlsx)",
        type=["xlsx"], key="tab7_upload",
    )

    if uploaded_raw is not None:
        with st.spinner("Scoring responses and computing psychometric metrics… (may take 30–60 s for IRT)"):
            try:
                import io
                Q_COLS_T7 = [f"Q{i:02d}" for i in range(1, 26)]

                raw = pd.read_excel(io.BytesIO(uploaded_raw.read()))

                # Score responses
                scored = raw.copy()
                for q in Q_COLS_T7:
                    scored[q] = raw[q].astype(str).str.strip().str.lower().map(
                        lambda x, k=ANSWER_KEY_T7[q]: 1.0 if x == k else (np.nan if x in ("nan", "") else 0.0)
                    )

                pre_s  = scored[scored["file_name"].str.contains("PRE",  case=False)].copy()
                post_s = scored[scored["file_name"].str.contains("POST", case=False)].copy()

                pre_diff  = pre_s[Q_COLS_T7].mean()
                post_diff = post_s[Q_COLS_T7].mean()
                gain_s    = (post_diff - pre_diff) / (1 - pre_diff)

                # CTT Discrimination (upper 27% − lower 27%)
                def _ctt_disc(sdf):
                    total = sdf[Q_COLS_T7].sum(axis=1)
                    cut = int(0.27 * len(sdf))
                    ui = total.nlargest(cut).index
                    li = total.nsmallest(cut).index
                    return pd.Series({q: sdf.loc[ui, q].mean() - sdf.loc[li, q].mean() for q in Q_COLS_T7})

                ctt_disc_s = _ctt_disc(pre_s)

                # Point-biserial
                pre_total = pre_s[Q_COLS_T7].sum(axis=1)
                pb_s = {}
                for q in Q_COLS_T7:
                    rest = pre_total - pre_s[q].fillna(0)
                    mask = pre_s[q].notna()
                    r, _ = pointbiserialr(pre_s.loc[mask, q], rest[mask])
                    pb_s[q] = r
                pb_series = pd.Series(pb_s)

                # Cronbach’s alpha + alpha-if-removed
                def _alpha(dfi):
                    dfi = dfi.dropna()
                    n = dfi.shape[1]
                    return (n / (n - 1)) * (1 - dfi.var(ddof=1).sum() / dfi.sum(axis=1).var(ddof=1))

                air = pd.Series({q: _alpha(pre_s[[c for c in Q_COLS_T7 if c != q]]) for q in Q_COLS_T7})

                # IRT 3PL via girth
                irt_a = pd.Series(np.nan, index=Q_COLS_T7)
                irt_b = pd.Series(np.nan, index=Q_COLS_T7)
                irt_c = pd.Series(np.nan, index=Q_COLS_T7)
                try:
                    from girth import threepl_mml
                    pre_complete = pre_s[Q_COLS_T7].dropna()
                    res = threepl_mml(pre_complete.values.T.astype(int))
                    irt_a = pd.Series(res["Discrimination"], index=Q_COLS_T7)
                    irt_b = pd.Series(res["Difficulty"],     index=Q_COLS_T7)
                    irt_c = pd.Series(res["Guessing"],       index=Q_COLS_T7)
                except Exception as irt_err:
                    st.warning(f"⚠️ IRT estimation skipped: {irt_err}. Columns will be NaN.")

                # Keep existing item type mapping
                existing_types = dict(zip(df_full["item"], df_full["type"]))

                result_df = pd.DataFrame({
                    "item":             Q_COLS_T7,
                    "type":             [existing_types.get(q, "E") for q in Q_COLS_T7],
                    "pre_test":         pre_diff.values,
                    "post_test":        post_diff.values,
                    "gain":             gain_s.values,
                    "ctt_diff":         pre_diff.values,
                    "ctt_disc":         ctt_disc_s.values,
                    "point_biserial":   pb_series.values,
                    "irt_diff":         irt_b.values,
                    "irt_disc":         irt_a.values,
                    "irt_guess":        irt_c.values,
                    "alpha_if_removed": air.values,
                }).round(4)

                st.success(
                    f"✅ Done! Scored {len(pre_s):,} PRE and {len(post_s):,} POST student records "
                    f"across {raw['file_name'].nunique()} semesters."
                )

                st.markdown("#### Generated Metrics Table")
                st.dataframe(result_df, use_container_width=True, hide_index=True)

                # Download button
                csv_out = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇ Download as CSV (import into Tab 4)",
                    data=csv_out,
                    file_name="computed_metrics.csv",
                    mime="text/csv",
                )

                st.info(
                    "💡 To apply this data to all charts, go to **Tab 4**, "
                    "open the **Upload CSV** section, and upload the downloaded file."
                )

            except Exception as e:
                st.error(f"An error occurred: {e}")
