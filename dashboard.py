import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

# ─── Filter ───────────────────────────────────────────────────────────────────
df = df_full[df_full["type_label"].isin(selected_types)].copy()

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
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Avg Post-Test</div>
    <div class="kpi-value">{avg_post:.2f}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Avg Norm. Gain</div>
    <div class="kpi-value">{avg_gain:.2f}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Problematic Items</div>
    <div class="kpi-value {'warn' if n_prob > 0 else ''}">{n_prob}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Negative Gain Items</div>
    <div class="kpi-value {'warn' if n_neg_gain > 0 else ''}">{n_neg_gain}</div>
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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Tab 1 — Discrimination vs Difficulty",
    "Tab 2 — Pre/Post & Gain",
    "Tab 3 — IRT Scatter",
    "Tab 4 — Full Metrics Table",
    "Tab 5 — Item Characteristic Curves",
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
    st.caption("Cells highlighted where values breach thresholds")

    display_cols = [
        "item", "type", "pre_test", "post_test", "gain",
        "ctt_diff", "ctt_disc", "point_biserial",
        "irt_diff", "irt_disc", "irt_guess", "alpha_if_removed",
    ]
    table_df = df[display_cols].copy()

    def style_table(df_s):
        styles = pd.DataFrame("", index=df_s.index, columns=df_s.columns)
        flagged = "background-color: #eeeeee; font-weight: 700; color: #000;"
        for idx in df_s.index:
            row = df_s.loc[idx]
            if row["ctt_diff"] < 0.20:
                styles.loc[idx, "ctt_diff"] = flagged
            if row["ctt_disc"] < 0.20:
                styles.loc[idx, "ctt_disc"] = flagged
            if row["point_biserial"] < 0.20:
                styles.loc[idx, "point_biserial"] = flagged
            if not (0.50 <= row["irt_disc"] <= 2.50):
                styles.loc[idx, "irt_disc"] = flagged
            if row["irt_guess"] > 0.25:
                styles.loc[idx, "irt_guess"] = flagged
            if row["alpha_if_removed"] > alpha_threshold:
                styles.loc[idx, "alpha_if_removed"] = flagged
            if row["gain"] < 0:
                styles.loc[idx, "gain"] = flagged
        return styles

    styled = (
        table_df.style
        .apply(style_table, axis=None)
        .format({
            "pre_test": "{:.3f}", "post_test": "{:.3f}", "gain": "{:.3f}",
            "ctt_diff": "{:.3f}", "ctt_disc": "{:.3f}", "point_biserial": "{:.3f}",
            "irt_diff": "{:.3f}", "irt_disc": "{:.3f}", "irt_guess": "{:.3f}",
            "alpha_if_removed": "{:.4f}",
        })
        .set_properties(**{"background-color": "#fff", "color": "#111", "border": "1px solid #e0e0e0"})
        .set_table_styles([
            {"selector": "th", "props": [
                ("background-color", "#f0f0f0"), ("color", "#111"),
                ("font-size", "11px"), ("text-transform", "uppercase"),
                ("border", "1px solid #ddd"), ("padding", "8px 10px"),
            ]},
            {"selector": "td", "props": [
                ("padding", "5px 10px"), ("font-size", "12px"),
            ]},
        ])
    )
    st.dataframe(styled, use_container_width=True, height=480)

    # ── CSV Download ──────────────────────────────────────────────────────────
    csv_bytes = table_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Download metrics as CSV",
        data=csv_bytes,
        file_name="emcs_metrics.csv",
        mime="text/csv",
    )

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
        "3PL logistic model: P(θ) = c + (1−c) / (1 + exp(−a·(θ−b)))  ·  "
        "Color = item type  ·  Dashed = flagged item"
    )

    theta = np.linspace(-4, 4, 300)

    fig5 = go.Figure()

    for _, row in df.iterrows():
        a, b, c = row["irt_disc"], row["irt_diff"], row["irt_guess"]
        prob  = c + (1 - c) / (1 + np.exp(-a * (theta - b)))
        color = TYPE_COLORS[row["type"]]
        dash  = "dash" if row["problematic"] else "solid"

        fig5.add_trace(go.Scatter(
            x=theta, y=prob,
            mode="lines",
            name=row["item"],
            line=dict(color=color, width=1.8, dash=dash),
            opacity=0.75,
            hovertemplate=(
                f"<b>{row['item']}</b><br>"
                f"a={a:.2f}, b={b:.2f}, c={c:.2f}<br>"
                "θ: %{x:.2f}<br>"
                "P(θ): %{y:.3f}<extra></extra>"
            ),
            showlegend=False,
        ))

    # Type + flagged legend entries
    for t_key, t_label in TYPE_LABELS.items():
        fig5.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines", name=t_label,
            line=dict(color=TYPE_COLORS[t_key], width=2.5),
            showlegend=True,
        ))
    fig5.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines", name="Flagged item",
        line=dict(color="#888", width=2, dash="dash"),
        showlegend=True,
    ))

    if show_thresholds:
        fig5.add_vline(x=0, line=dict(color="#bbb", width=1, dash="dot"))
        fig5.add_hline(y=0.5, line=dict(color="#bbb", width=1, dash="dot"))

    fig5.update_layout(
        **LAYOUT_BASE,
        legend=LEGEND_STYLE,
        xaxis=dict(title="Ability (θ)", range=[-4, 4]),
        yaxis=dict(title="Probability of Correct Response P(θ)", range=[0, 1.02]),
        height=560,
    )
    clean_axes(fig5)
    st.plotly_chart(fig5, use_container_width=True)
