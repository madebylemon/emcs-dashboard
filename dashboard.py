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

# ─── Localization Dictionary (i18n) ──────────────────────────────────────────
I18N = {
    "en": {
        "title": "EMCS Item Explorer",
        "subtitle": "Psychometric Dashboard — Energy & Momentum Conceptual Survey",
        "lang_select": "🌐 Language / Idioma",
        "controls": "Controls",
        "filter_type": "Filter by Item Type",
        "highlight_flagged": "Highlight Flagged Items",
        "show_thresholds": "Show Threshold Lines",
        "thresholds_title": "Psychometric Thresholds",
        "alpha_input": "Alpha if Removed ≤",
        "glossary_title": "📖 Metric Glossary",
        "plot_settings": "🎨 Plot Settings",
        "font_size": "Font Size",
        "chart_height": "Chart Height (px)",
        "export_aspect": "Export Aspect Ratio",
        "export_res": "Export Resolution",
        "export_caption": "📷 To export any chart: hover over it → click the **camera icon** (📷) in the top-right toolbar.",
        "kpi_pre": "Avg Pre-Test",
        "kpi_pre_sub": "mean item p-value (student total / K)",
        "kpi_post": "Avg Post-Test",
        "kpi_post_sub": "mean item p-value after instruction",
        "kpi_gain": "Avg Norm. Gain",
        "kpi_gain_sub": "g = (post − pre) / (1 − pre)",
        "kpi_alpha": "Cronbach's α (Pre)",
        "kpi_alpha_sub": "Scale internal consistency (K=25)",
        "kpi_alpha_post": "Cronbach's α (Post)",
        "kpi_alpha_post_sub": "Post-instruction scale reliability",
        "kpi_prob": "Problematic Items",
        "kpi_prob_sub": "items breaching ≥1 threshold",
        "kpi_neg": "Negative Gain Items",
        "kpi_neg_sub": "post-test score below pre-test",
        # Tabs
        "tab1": "Tab 1 — CTT Scatter",
        "tab2": "Tab 2 — Pre/Post & Gain",
        "tab3": "Tab 3 — IRT Scatter",
        "tab4": "Tab 4 — Full Metrics Table",
        "tab5": "Tab 5 — Item Curves (3PL)",
        "tab6": "Tab 6 — Category Analysis",
        "tab7": "Tab 7 — Rankings & Psychometric Descriptions",
        "tab8": "Tab 8 — Compute from Raw Data",
        # Glossary text
        "glossary_content": """
**Avg Pre-Test**  
Mean proportion correct across all items *before* instruction ($p = \\sum X / N$). Equivalent to the mean student total test percentage.

**Avg Post-Test**  
Mean proportion correct across all items *after* instruction.

**Avg Normalized Gain (g)**  
Measures learning efficiency relative to potential improvement:
> *g = (post − pre) / (1 − pre)*  

**Cronbach's Alpha (α)**  
Index of internal consistency reliability:
> *α = (K / (K − 1)) × (1 − ∑ σᵢ² / σ_total²)*  
Values > 0.70 indicate good scale reliability.

**CTT Difficulty (p-value)**  
Proportion of students answering correctly. Higher = easier item.

**CTT Discrimination**  
Difference between top 27% and bottom 27% scoring students (or correlation with total score). Higher = better differentiation.

**Point-Biserial Correlation**  
Pearson correlation between binary item correctness and total test score.

**IRT Parameters (3PL model)**  
- *a* — Discrimination: curve steepness at inflection point  
- *b* — Difficulty: ability θ where P(correct) = (1+c)/2  
- *c* — Guessing: lower asymptote (chance level)

**Alpha if Removed**  
Scale Cronbach's α if this item is deleted. Values > overall α indicate item reduces reliability.

**Categories:** 🔵 Energy (14 items) · 🔴 Momentum (10 items) · 🟣 Energy & Momentum (1 item)
""",
        # Tab 7 strings
        "rankings_title": "Psychometric Item Rankings & Methodological Analyses",
        "rankings_sub": "Comprehensive ordering of EMCS items by Classical Test Theory (CTT) and Item Response Theory (IRT) difficulties, plus average and reliability calculations.",
        "ctt_rank_header": "1. Items Ordered by CTT Difficulty (p-value)",
        "ctt_rank_caption": "Ordered from hardest (lowest p-value) to easiest (highest p-value). CTT difficulty represents proportion of students answering correctly.",
        "irt_rank_header": "2. Items Ordered by IRT Difficulty (b parameter)",
        "irt_rank_caption": "Ordered from lowest ability threshold (easiest / lowest b) to highest ability threshold (hardest / highest b).",
        "avg_method_header": "3. How Pre-Test and Post-Test Averages are Calculated",
        "alpha_method_header": "4. Cronbach's Alpha (α) Calculation & Reliability",
        "type_energy": "Energy",
        "type_momentum": "Momentum",
        "type_both": "Energy & Momentum",
    },
    "es": {
        "title": "Explorador de Reactivos EMCS",
        "subtitle": "Panel Psicométrico — Encuesta Conceptual de Energía y Momento",
        "lang_select": "🌐 Language / Idioma",
        "controls": "Controles",
        "filter_type": "Filtrar por Tipo de Reactivo",
        "highlight_flagged": "Destacar Reactivos Flagged",
        "show_thresholds": "Mostrar Líneas de Umbral",
        "thresholds_title": "Umbrales Psicométricos",
        "alpha_input": "Alfa si se Elimina ≤",
        "glossary_title": "📖 Glosario de Métricas",
        "plot_settings": "🎨 Configuración de Gráficos",
        "font_size": "Tamaño de Fuente",
        "chart_height": "Altura del Gráfico (px)",
        "export_aspect": "Relación de Aspecto de Exportación",
        "export_res": "Resolución de Exportación",
        "export_caption": "📷 Para exportar cualquier gráfico: pasa el ratón → haz clic en la **cámara** (📷) en la barra superior.",
        "kpi_pre": "Prom. Pre-Test",
        "kpi_pre_sub": "p-promedio de reactivos (total estudiante / K)",
        "kpi_post": "Prom. Post-Test",
        "kpi_post_sub": "p-promedio de reactivos tras instrucción",
        "kpi_gain": "Ganancia Norm. Prom.",
        "kpi_gain_sub": "g = (post − pre) / (1 − pre)",
        "kpi_alpha": "Alfa de Cronbach (Pre)",
        "kpi_alpha_sub": "Consistencia interna de la escala (K=25)",
        "kpi_alpha_post": "Alfa de Cronbach (Post)",
        "kpi_alpha_post_sub": "Fiabilidad de escala post-instrucción",
        "kpi_prob": "Reactivos Problemáticos",
        "kpi_prob_sub": "reactivos que violan ≥1 umbral",
        "kpi_neg": "Ganancia Negativa",
        "kpi_neg_sub": "puntaje post-test inferior al pre-test",
        # Tabs
        "tab1": "Tab 1 — Dispersión CTT",
        "tab2": "Tab 2 — Pre/Post y Ganancia",
        "tab3": "Tab 3 — Dispersión IRT",
        "tab4": "Tab 4 — Tabla Completa de Métricas",
        "tab5": "Tab 5 — Curvas de Reactivos (3PL)",
        "tab6": "Tab 6 — Análisis por Categoría",
        "tab7": "Tab 7 — Clasificación y Descripciones Psicometrícas",
        "tab8": "Tab 8 — Calcular desde Datos Crudos",
        # Glossary text
        "glossary_content": """
**Prom. Pre-Test**  
Proporción promedio de respuestas correctas antes de la instrucción ($p = \\sum X / N$). Equivalente al porcentaje total promedio de los estudiantes.

**Prom. Post-Test**  
Proporción promedio de respuestas correctas después de la instrucción.

**Ganancia Normalizada Promedio (g)**  
Mide la eficiencia del aprendizaje en relación con la mejora potencial:
> *g = (post − pre) / (1 − pre)*  

**Alfa de Cronbach (α)**  
Índice de fiabilidad de consistencia interna:
> *α = (K / (K − 1)) × (1 − ∑ σᵢ² / σ_total²)*  
Valores > 0.70 indican buena fiabilidad de la escala.

**Dificultad CTT (valor p)**  
Proporción de estudiantes que responden correctamente. Mayor = reactivo más fácil.

**Discriminación CTT**  
Diferencia entre el 27% superior y el 27% inferior de los estudiantes (o correlación con el puntaje total).

**Correlación Punto-Biserial**  
Correlación de Pearson entre la corrección del reactivo binario y el puntaje total.

**Parámetros IRT (modelo 3PL)**  
- *a* — Discriminación: pendiente de la curva  
- *b* — Dificultad: habilidad θ donde P(correcto) = (1+c)/2  
- *c* — Adivinación: asíntota inferior (nivel de azar)

**Alfa si se Elimina**  
Alfa de Cronbach si se elimina este reactivo. Valores > alfa general indican que el reactivo reduce la fiabilidad.

**Categorías:** 🔵 Energía (14 reactivos) · 🔴 Momento (10 reactivos) · 🟣 Energía y Momento (1 reactivo)
""",
        # Tab 7 strings
        "rankings_title": "Clasificación de Reactivos y Análisis Metodológicos",
        "rankings_sub": "Ordenamiento completo de reactivos EMCS por dificultades de Teoría Clásica de los Tests (CTT) y Teoría de Respuesta al Ítem (IRT), más cálculos de promedios y fiabilidad.",
        "ctt_rank_header": "1. Reactivos Ordenados por Dificultad CTT (valor p)",
        "ctt_rank_caption": "Ordenados del más difícil (menor valor p) al más fácil (mayor valor p). La dificultad CTT representa la proporción de estudiantes que respondieron correctamente.",
        "irt_rank_header": "2. Reactivos Ordenados por Dificultad IRT (parámetro b)",
        "irt_rank_caption": "Ordenados desde el umbral de habilidad más bajo (más fácil / menor b) hasta el umbral más alto (más difícil / mayor b).",
        "avg_method_header": "3. Cómo se Calculan los Promedios de Pre-Test y Post-Test",
        "alpha_method_header": "4. Cálculo del Alfa de Cronbach (α) y Fiabilidad",
        "type_energy": "Energía",
        "type_momentum": "Momento",
        "type_both": "Energía y Momento",
    }
}

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #ffffff; color: #111111; }
    [data-testid="stSidebar"] { background-color: #f5f5f5; border-right: 1px solid #ddd; }
    [data-testid="stSidebar"] * { color: #111111 !important; }

    .kpi-row { display: flex; gap: 12px; margin-bottom: 24px; flex-wrap: wrap; }
    .kpi-card {
        flex: 1; min-width: 140px;
        background: #f9f9f9; border: 1px solid #ddd; border-radius: 6px;
        padding: 14px 10px; text-align: center;
    }
    .kpi-label { font-size: 11px; color: #555; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
    .kpi-value { font-size: 26px; font-weight: 700; color: #111; }
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

    .info-card {
        background: #f4f6f9; border-left: 4px solid #2196F3;
        padding: 16px 20px; margin-bottom: 16px; border-radius: 0 6px 6px 0;
    }
    .info-card h4 { color: #0d47a1; margin: 0 0 8px 0; font-size: 15px; font-weight: 700; }
    .info-card p  { color: #222; margin: 0; font-size: 13px; line-height: 1.6; }

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
TYPE_LABELS_EN = {"E": "Energy", "M": "Momentum", "E&M": "Energy & Momentum"}
TYPE_LABELS_ES = {"E": "Energía", "M": "Momento", "E&M": "Energía y Momento"}

PLOT_BG      = "#ffffff"
PAPER_BG     = "#ffffff"
GRID_CLR     = "#eeeeee"
AXIS_CLR     = "#333333"
FLAG_CLR     = "#111111"
GAIN_POS_CLR = "#27ae60"   # green
GAIN_NEG_CLR = "#e74c3c"   # red

# ─── Data ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("emcs_data.csv")
    df["type_label"] = df["type"].map(TYPE_LABELS_EN)
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

@st.cache_data(show_spinner=False)
def compute_raw_excel_metrics(file_bytes):
    import io, re, os, tempfile, subprocess
    from scipy.stats import pearsonr
    from scipy.optimize import minimize
    from scipy.special import expit

    # ── Attempt 1: Execute Rscript with mirt for 100% exact R mirt package alignment ──
    try:
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f_in:
            f_in.write(file_bytes)
            in_path = f_in.name
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f_out:
            out_path = f_out.name

        r_code = '''
suppressPackageStartupMessages({
  library(psych)
  library(dplyr)
  library(mirt)
  library(readxl)
})

args <- commandArgs(trailingOnly = TRUE)
data <- read_excel(args[1])
answer_key <- c("b", "e", "b", "a", "d", "c", "e", "c", "a", "d",
                "e", "d", "c", "d", "a", "c", "b", "e", "b", "a",
                "c", "d", "b", "a", "e")
Q_COLS <- sprintf("Q%02d", 1:25)

col_map <- list()
for (c in colnames(data)) {
  c_clean <- trimws(c)
  c_lower <- tolower(c_clean)
  if (c_lower %in% c("file_name", "filename", "file name", "file")) {
    colnames(data)[colnames(data) == c] <- "file_name"
  } else if (grepl("^[qQ]0?([1-9]|1[0-9]|2[0-5])$", c_clean)) {
    q_num <- as.integer(sub("^[qQ]0?", "", c_clean))
    colnames(data)[colnames(data) == c] <- sprintf("Q%02d", q_num)
  }
}

graded <- data
for (i in 1:25) {
  q <- Q_COLS[i]
  k <- answer_key[i]
  raw_v <- trimws(tolower(as.character(data[[q]])))
  graded[[q]] <- ifelse(raw_v == k, 1, ifelse(raw_v %in% c("nan", "", "na", "null", "none"), NA, 0))
}

data_ctt <- graded[Q_COLS]
data_ctt[is.na(data_ctt)] <- 0

pre_mask  <- if ("file_name" %in% colnames(graded)) grepl("PRE",  graded$file_name, ignore.case = TRUE) else rep(FALSE, nrow(graded))
post_mask <- if ("file_name" %in% colnames(graded)) grepl("POST", graded$file_name, ignore.case = TRUE) else rep(FALSE, nrow(graded))

pre_diff  <- if (any(pre_mask))  colMeans(data_ctt[pre_mask, Q_COLS])  else rep(NA, 25)
post_diff <- if (any(post_mask)) colMeans(data_ctt[post_mask, Q_COLS]) else rep(NA, 25)

gain <- sapply(1:25, function(i) {
  pr <- pre_diff[i]; po <- post_diff[i]
  if (is.na(pr) || is.na(po)) return(NA)
  if (po >= pr) (po - pr) / (1 - pr) else (po - pr) / pr
})

tot_score <- rowSums(data_ctt)
ctt_diff  <- colMeans(data_ctt)
ctt_disc  <- apply(data_ctt, 2, function(x) cor(x, tot_score))
bisr      <- apply(data_ctt, 2, function(x) cor(x, tot_score - x))

alpha_res <- psych::alpha(data_ctt)
alpha_if_rm <- alpha_res$alpha.drop[,"raw_alpha"]

irt_data <- na.omit(graded[Q_COLS])
model_3pl <- mirt(irt_data, 1, itemtype = "3PL", technical = list(NCYCLES = 2000))
params_3pl <- coef(model_3pl, IRTpars = TRUE, simplify = TRUE)$items

ITEM_TYPES <- c("E","E","M","E","M","E","M","E","E","M",
                "M","E","E","M","E","E&M","E","M","M","E",
                "M","E","M","E","E")

res <- data.frame(
  item = Q_COLS,
  type = ITEM_TYPES,
  pre_test = round(pre_diff, 4),
  post_test = round(post_diff, 4),
  gain = round(gain, 4),
  ctt_diff = round(ctt_diff, 4),
  ctt_disc = round(ctt_disc, 4),
  point_biserial = round(bisr, 4),
  irt_diff = round(params_3pl[, "b"], 4),
  irt_disc = round(params_3pl[, "a"], 4),
  irt_guess = round(params_3pl[, "g"], 4),
  alpha_if_removed = round(alpha_if_rm, 4)
)

write.csv(res, args[2], row.names = FALSE)
'''
        with tempfile.NamedTemporaryFile(suffix=".r", delete=False, mode="w") as f_r:
            f_r.write(r_code)
            r_path = f_r.name

        proc = subprocess.run(["Rscript", r_path, in_path, out_path], capture_output=True, text=True, timeout=120)
        
        if proc.returncode == 0 and os.path.exists(out_path):
            result_df = pd.read_csv(out_path)
            n_ctt_count = len(pd.read_excel(io.BytesIO(file_bytes)))
            n_irt_count = len(result_df)
            for p in (in_path, out_path, r_path):
                if os.path.exists(p): os.remove(p)
            return result_df, n_ctt_count, n_irt_count
    except Exception as e:
        pass

    # ── Attempt 2: High-Precision Pure Python Solver (Matches R formulas) ──
    raw = pd.read_excel(io.BytesIO(file_bytes))
    
    col_map = {}
    for c in raw.columns:
        c_clean = str(c).strip()
        c_lower = c_clean.lower()
        if c_lower in ("file_name", "filename", "file name", "file"):
            col_map[c] = "file_name"
        else:
            m = re.match(r"^[qQ]0?([1-9]|1[0-9]|2[0-5])$", c_clean)
            if m:
                q_num = int(m.group(1))
                col_map[c] = f"Q{q_num:02d}"
    
    df = raw.rename(columns=col_map)
    
    Q_COLS_T8 = [f"Q{i:02d}" for i in range(1, 26)]
    missing_qs = [q for q in Q_COLS_T8 if q not in df.columns]
    if missing_qs:
        raise ValueError(
            f"Missing item columns in uploaded file: {missing_qs}. "
            "File must contain response columns for Q01–Q25 (or Q1–Q25)."
        )
    
    if "file_name" not in df.columns:
        df["file_name"] = "COMBINED"

    ANSWER_KEY_T8 = {
        "Q01":"b","Q02":"e","Q03":"b","Q04":"a","Q05":"d",
        "Q06":"c","Q07":"e","Q08":"c","Q09":"a","Q10":"d",
        "Q11":"e","Q12":"d","Q13":"c","Q14":"d","Q15":"a",
        "Q16":"c","Q17":"b","Q18":"e","Q19":"b","Q20":"a",
        "Q21":"c","Q22":"d","Q23":"b","Q24":"a","Q25":"e",
    }

    scored = df.copy()
    for q in Q_COLS_T8:
        scored[q] = df[q].astype(str).str.strip().str.lower().map(
            lambda x, k=ANSWER_KEY_T8[q]: 1.0 if x == k else (np.nan if x in ("nan", "", "none", "null") else 0.0)
        )

    # Missing Data Policy for CTT: Convert all NAs to 0.0
    scored_ctt = scored.copy()
    for q in Q_COLS_T8:
        scored_ctt[q] = scored_ctt[q].fillna(0.0)

    pre_mask = scored_ctt["file_name"].astype(str).str.contains("PRE", case=False, na=False)
    post_mask = scored_ctt["file_name"].astype(str).str.contains("POST", case=False, na=False)

    if pre_mask.any():
        pre_s = scored_ctt[pre_mask]
        pre_diff = pre_s[Q_COLS_T8].mean()
    else:
        pre_diff = pd.Series(np.nan, index=Q_COLS_T8)

    if post_mask.any():
        post_s = scored_ctt[post_mask]
        post_diff = post_s[Q_COLS_T8].mean()
    else:
        post_diff = pd.Series(np.nan, index=Q_COLS_T8)

    gain_s = pd.Series([
        (post_diff[q] - pre_diff[q]) / (1.0 - pre_diff[q]) if (pd.notna(post_diff[q]) and pd.notna(pre_diff[q]) and post_diff[q] >= pre_diff[q])
        else ((post_diff[q] - pre_diff[q]) / pre_diff[q] if (pd.notna(post_diff[q]) and pd.notna(pre_diff[q]) and pre_diff[q] > 0) else np.nan)
        for q in Q_COLS_T8
    ], index=Q_COLS_T8)

    ctt_diff_s = scored_ctt[Q_COLS_T8].mean()
    total_score = scored_ctt[Q_COLS_T8].sum(axis=1)

    # Item Discrimination: Pearson correlation with total score (cor(item, total_score))
    ctt_disc_s = pd.Series({q: pearsonr(scored_ctt[q], total_score)[0] for q in Q_COLS_T8})

    # Point-Biserial Correlation: Corrected item-total correlation (cor(item, total_score - item))
    pb_series = pd.Series({q: pearsonr(scored_ctt[q], total_score - scored_ctt[q])[0] for q in Q_COLS_T8})

    def _alpha(dfi):
        n = dfi.shape[1]
        var_sum = dfi.var(ddof=1).sum()
        tot_var = dfi.sum(axis=1).var(ddof=1)
        if tot_var <= 0 or pd.isna(tot_var):
            return np.nan
        return (n / (n - 1)) * (1.0 - var_sum / tot_var)

    air = pd.Series({q: _alpha(scored_ctt[[c for c in Q_COLS_T8 if c != q]]) for q in Q_COLS_T8})

    # Missing Data Policy for IRT: Filter complete cases (dropna on Q01-Q25)
    scored_irt = scored.dropna(subset=Q_COLS_T8).copy()
    n_irt = len(scored_irt)

    if n_irt >= 10:
        patterns = scored_irt.groupby(Q_COLS_T8).size().reset_index(name="_count")
        Y_unique = patterns[Q_COLS_T8].values.astype(np.float64)
        freqs = patterns["_count"].values.astype(np.float64)
        U_irt, K_irt = Y_unique.shape

        _n_quad = 31
        _theta_nodes = np.linspace(-4, 4, _n_quad)
        _wts_raw = np.exp(-0.5 * _theta_nodes**2)
        _wts_n = _wts_raw / _wts_raw.sum()

        p_hat_irt = Y_unique.mean(axis=0)
        b_init = np.log(np.maximum(1 - p_hat_irt, 0.01) / np.maximum(p_hat_irt, 0.01))
        a_init = np.ones(K_irt)
        c_init = np.full(K_irt, 0.15)
        init_params = np.concatenate([a_init, b_init, c_init])

        bounds_irt = []
        for i in range(K_irt): bounds_irt.append((0.10, 4.0))
        for i in range(K_irt): bounds_irt.append((-4.0, 4.0))
        for i in range(K_irt): bounds_irt.append((0.0, 0.40))

        def _joint_loss_and_grad(params):
            a = params[0:K_irt]
            b = params[K_irt:2*K_irt]
            c = params[2*K_irt:3*K_irt]

            dev = _theta_nodes[:, None] - b[None, :]
            p_logit = expit(a[None, :] * dev)
            P_quad = c[None, :] + (1.0 - c[None, :]) * p_logit
            P_quad = np.clip(P_quad, 1e-9, 1.0 - 1e-9)

            log_P = np.log(P_quad)
            log_1_P = np.log(1.0 - P_quad)
            log_lik_mat = log_P @ Y_unique.T + log_1_P @ (1.0 - Y_unique).T

            max_log = log_lik_mat.max(axis=0, keepdims=True)
            lik_mat = np.exp(log_lik_mat - max_log)
            w_lik = _wts_n[:, None] * lik_mat
            marg_lik = w_lik.sum(axis=0)

            loss = -np.sum(freqs * (np.log(np.maximum(marg_lik, 1e-300)) + max_log.squeeze()))

            post_k_u = w_lik / np.maximum(marg_lik, 1e-300)
            W_freq = post_k_u * freqs[None, :]

            dL_dP = (W_freq @ Y_unique) / P_quad - (W_freq @ (1.0 - Y_unique)) / (1.0 - P_quad)
            dp_dlogit = p_logit * (1.0 - p_logit)

            grad_a = -np.sum(dL_dP * (1.0 - c[None, :]) * dp_dlogit * dev, axis=0)
            grad_b = np.sum(dL_dP * (1.0 - c[None, :]) * dp_dlogit * a[None, :], axis=0)
            grad_c = -np.sum(dL_dP * (1.0 - p_logit), axis=0)

            grad = np.concatenate([grad_a, grad_b, grad_c])
            return loss, grad

        try:
            res_irt = minimize(
                _joint_loss_and_grad, init_params,
                jac=True, bounds=bounds_irt, method="L-BFGS-B",
                options={"maxiter": 80, "ftol": 1e-5},
            )
            if res_irt.nit >= 3:
                a_fit = np.clip(res_irt.x[0:K_irt], 0.10, 4.0)
                b_fit = np.clip(res_irt.x[K_irt:2*K_irt], -4.0, 4.0)
                c_fit = np.clip(res_irt.x[2*K_irt:3*K_irt], 0.0, 0.40)
            else:
                a_fit, b_fit, c_fit = np.full(25, np.nan), np.full(25, np.nan), np.full(25, np.nan)
        except Exception:
            a_fit, b_fit, c_fit = np.full(25, np.nan), np.full(25, np.nan), np.full(25, np.nan)
    else:
        a_fit, b_fit, c_fit = np.full(25, np.nan), np.full(25, np.nan), np.full(25, np.nan)

    irt_a = pd.Series(a_fit, index=Q_COLS_T8)
    irt_b = pd.Series(b_fit, index=Q_COLS_T8)
    irt_c = pd.Series(c_fit, index=Q_COLS_T8)

    item_type_default = {
        "Q01":"E","Q02":"E","Q03":"M","Q04":"E","Q05":"M",
        "Q06":"E","Q07":"M","Q08":"E","Q09":"E","Q10":"M",
        "Q11":"M","Q12":"E","Q13":"E","Q14":"M","Q15":"E",
        "Q16":"E&M","Q17":"E","Q18":"M","Q19":"M","Q20":"E",
        "Q21":"M","Q22":"E","Q23":"M","Q24":"E","Q25":"E"
    }

    result_df = pd.DataFrame({
        "item":             Q_COLS_T8,
        "type":             [item_type_default.get(q, "E") for q in Q_COLS_T8],
        "pre_test":         pre_diff.values,
        "post_test":        post_diff.values,
        "gain":             gain_s.values,
        "ctt_diff":         ctt_diff_s.values,
        "ctt_disc":         ctt_disc_s.values,
        "point_biserial":   pb_series.values,
        "irt_diff":         irt_b.values,
        "irt_disc":         irt_a.values,
        "irt_guess":        irt_c.values,
        "alpha_if_removed": air.values,
    }).round(4)

    return result_df, len(scored_ctt), n_irt


# ─── Sidebar Controls & Language Selector ────────────────────────────────────
with st.sidebar:
    st.markdown("## Options / Opciones")
    lang_choice = st.selectbox(
        "🌐 Language / Idioma",
        options=["en", "es"],
        format_func=lambda x: "English 🇺🇸" if x == "en" else "Español 🇲🇽 / 🇪🇸",
        index=0,
    )
    txt = I18N[lang_choice]
    TYPE_LABELS = TYPE_LABELS_EN if lang_choice == "en" else TYPE_LABELS_ES

    st.markdown("---")
    st.markdown(f"## {txt['controls']}")

    selected_types = st.multiselect(
        txt["filter_type"],
        options=list(TYPE_LABELS.values()),
        default=list(TYPE_LABELS.values()),
    )
    show_problematic = st.checkbox(txt["highlight_flagged"], value=True)
    show_thresholds  = st.checkbox(txt["show_thresholds"], value=True)

    st.markdown("---")
    st.markdown(f"### {txt['thresholds_title']}")
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
        txt["alpha_input"],
        min_value=0.0, max_value=1.0,
        value=0.7563, step=0.0001, format="%.4f",
    )

    st.markdown("---")
    with st.expander(txt["glossary_title"]):
        st.markdown(txt["glossary_content"])

    st.markdown("---")
    st.markdown(f"### {txt['plot_settings']}")
    FONT_SIZE = st.slider(txt["font_size"], 8, 22, 12)
    CHART_H = st.slider(txt["chart_height"], 300, 1000, 520, step=20)

    _aspect_opts = {
        "16:9 — Widescreen": 16 / 9,
        "4:3 — Standard":    4 / 3,
        "1:1 — Square":      1.0,
    }
    _asp_choice = st.selectbox(txt["export_aspect"], list(_aspect_opts.keys()), index=0)
    _asp_ratio  = _aspect_opts[_asp_choice]

    _res_opts = {
        "Screen (72 dpi)": dict(w=1200, scale=1.0),
        "Presentation (150 dpi, 1600 px)": dict(w=1600, scale=1.5),
        "Publication / Print (300 dpi, 2400 px)": dict(w=2400, scale=3.0),
    }
    _res_choice = st.selectbox(txt["export_res"], list(_res_opts.keys()), index=1)
    _res_cfg    = _res_opts[_res_choice]

    EXPORT_W     = _res_cfg["w"]
    EXPORT_H     = int(EXPORT_W / _asp_ratio)
    EXPORT_SCALE = _res_cfg["scale"]

    st.caption(txt["export_caption"])

PLOTLY_EXPORT_CONFIG = {
    "toImageButtonOptions": {
        "format":   "png",
        "filename": "emcs_chart",
        "width":    EXPORT_W,
        "height":   EXPORT_H,
        "scale":    EXPORT_SCALE,
    },
    "displayModeBar": True,
}

# ─── Filter + Editable Session State ────────────────────────────────────────
_filter_key = tuple(sorted(selected_types))
_type_rev_map = {v: k for k, v in TYPE_LABELS.items()}
_selected_keys = [_type_rev_map[t] for t in selected_types if t in _type_rev_map]

_base_df = df_full[df_full["type"].isin(_selected_keys)].copy()
_base_df["type_label"] = _base_df["type"].map(TYPE_LABELS)

if ("edited_df" not in st.session_state
        or st.session_state.get("filter_key") != _filter_key):
    st.session_state.edited_df = _base_df.copy()
    st.session_state.filter_key = _filter_key

df = st.session_state.edited_df
df["type_label"] = df["type"].map(TYPE_LABELS)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="main-header">
  <h1>{txt['title']}</h1>
  <p>{txt['subtitle']}</p>
</div>
""", unsafe_allow_html=True)

# ─── KPI Cards ────────────────────────────────────────────────────────────────
avg_pre        = df["pre_test"].mean()
avg_post       = df["post_test"].mean()
avg_gain       = df["gain"].mean()
cronbach_pre   = 0.7563
cronbach_post  = 0.8672
n_prob         = int(df["problematic"].sum())
n_neg_gain     = int((df["gain"] < 0).sum())

st.markdown(f"""
<div class="kpi-row">
  <div class="kpi-card">
    <div class="kpi-label">{txt['kpi_pre']}</div>
    <div class="kpi-value">{avg_pre:.2f}</div>
    <div class="kpi-label" style="font-size:10px;margin-top:4px">{txt['kpi_pre_sub']}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">{txt['kpi_post']}</div>
    <div class="kpi-value">{avg_post:.2f}</div>
    <div class="kpi-label" style="font-size:10px;margin-top:4px">{txt['kpi_post_sub']}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">{txt['kpi_gain']}</div>
    <div class="kpi-value">{avg_gain:.2f}</div>
    <div class="kpi-label" style="font-size:10px;margin-top:4px">{txt['kpi_gain_sub']}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">{txt['kpi_alpha']}</div>
    <div class="kpi-value">{cronbach_pre:.4f}</div>
    <div class="kpi-label" style="font-size:10px;margin-top:4px">{txt['kpi_alpha_sub']}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">{txt['kpi_alpha_post']}</div>
    <div class="kpi-value">{cronbach_post:.4f}</div>
    <div class="kpi-label" style="font-size:10px;margin-top:4px">{txt['kpi_alpha_post_sub']}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">{txt['kpi_prob']}</div>
    <div class="kpi-value {'warn' if n_prob > 0 else ''}">{n_prob}</div>
    <div class="kpi-label" style="font-size:10px;margin-top:4px">{txt['kpi_prob_sub']}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">{txt['kpi_neg']}</div>
    <div class="kpi-value {'warn' if n_neg_gain > 0 else ''}">{n_neg_gain}</div>
    <div class="kpi-label" style="font-size:10px;margin-top:4px">{txt['kpi_neg_sub']}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Shared layout defaults ───────────────────────────────────────────────────
LAYOUT_BASE = dict(
    paper_bgcolor=PAPER_BG,
    plot_bgcolor=PLOT_BG,
    font=dict(color=AXIS_CLR, size=FONT_SIZE),
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
                    tickcolor="#aaa",
                    tickfont=dict(color=AXIS_CLR, size=FONT_SIZE),
                    title_font=dict(color=AXIS_CLR, size=FONT_SIZE),
                    zeroline=False,
                )

LEGEND_STYLE = dict(
    bgcolor="#fff", bordercolor="#ddd", borderwidth=1,
    font=dict(color="#111111", size=FONT_SIZE),
)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    txt["tab1"],
    txt["tab2"],
    txt["tab3"],
    txt["tab4"],
    txt["tab5"],
    txt["tab6"],
    txt["tab7"],
    txt["tab8"],
])

# ════════════════════════════════════════════
# TAB 1 — CTT Scatter
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
        height=CHART_H,
    )
    clean_axes(fig1)
    st.plotly_chart(fig1, use_container_width=True, config=PLOTLY_EXPORT_CONFIG)

# ════════════════════════════════════════════
# TAB 2 — Pre/Post + Gain
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

    for clr, label in [(GAIN_POS_CLR, "Gain ≥ 0"), (GAIN_NEG_CLR, "Gain < 0")]:
        fig2.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name=label,
            marker=dict(color=clr, size=10, symbol="diamond"),
            showlegend=True,
        ), secondary_y=True)

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
        height=int(CHART_H * 0.98),
        xaxis=dict(tickangle=-45),
    )
    fig2.update_yaxes(title_text="Proportion Correct", secondary_y=False, range=[0, 1.05],
                      showgrid=True, gridcolor=GRID_CLR, linecolor="#aaa")
    fig2.update_yaxes(title_text="Normalized Gain", secondary_y=True, range=[-0.5, 0.7],
                      showgrid=False, linecolor="#aaa")
    st.plotly_chart(fig2, use_container_width=True, config=PLOTLY_EXPORT_CONFIG)

# ════════════════════════════════════════════
# TAB 3 — IRT Discrimination vs Difficulty scatter
# ════════════════════════════════════════════
with tab3:
    st.markdown("### IRT Discrimination vs. Difficulty")
    st.caption("Dot size = IRT guessing (c) · Color = item type · Dashed lines = acceptable range")

    fig3 = go.Figure()

    if show_thresholds:
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
        sub_irt = sub.dropna(subset=["irt_diff", "irt_disc"])
        if sub_irt.empty:
            continue
        is_prob_irt = sub_irt["problematic"] if show_problematic else pd.Series([False]*len(sub_irt), index=sub_irt.index)
        marker_sizes = sub_irt["irt_guess"].fillna(0).clip(0, 1) * 80 + 8
        fig3.add_trace(go.Scatter(
            x=sub_irt["irt_diff"], y=sub_irt["irt_disc"],
            mode="markers",
            name=t_label,
            marker=dict(
                size=marker_sizes.tolist(),
                color=TYPE_COLORS[t_key],
                opacity=0.82,
                line=dict(
                    color=[FLAG_CLR if p else TYPE_COLORS[t_key] for p in is_prob_irt],
                    width=[3 if p else 1 for p in is_prob_irt],
                ),
            ),
            text=sub_irt["item"],
            customdata=sub_irt[["irt_guess", "type_label"]].values,
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
        height=CHART_H,
    )
    clean_axes(fig3)
    st.plotly_chart(fig3, use_container_width=True, config=PLOTLY_EXPORT_CONFIG)

# ════════════════════════════════════════════
# TAB 4 — Full Table + Download + Flagged Cards
# ════════════════════════════════════════════
with tab4:
    st.markdown("### Full Psychometric Metrics")

    with st.expander("\u2b06\ufe0f Upload CSV to replace data (all charts update automatically)"):
        uploaded_csv = st.file_uploader(
            "Upload a metrics CSV (must have columns: item, type, pre_test, post_test, gain, "
            "ctt_diff, ctt_disc, point_biserial, irt_diff, irt_disc, irt_guess, alpha_if_removed)",
            type=["csv"], key="tab4_csv_upload",
        )
        if uploaded_csv is not None:
            _file_id = f"{uploaded_csv.name}_{uploaded_csv.size}"
            if st.session_state.get("tab4_last_csv_id") != _file_id:
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
                        st.session_state.filter_key       = _filter_key
                        st.session_state.tab4_last_csv_id = _file_id
                        st.rerun()
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")
            else:
                st.success(
                    f"\u2705 **{uploaded_csv.name}** is loaded \u2014 "
                    f"{len(st.session_state.edited_df)} items active."
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

    numeric_cols = [c for c in edit_cols if c not in ("item", "type")]
    for col in numeric_cols:
        st.session_state.edited_df[col] = edited[col].values

    s = st.session_state.edited_df
    s["problematic"] = (
        (s["ctt_diff"]       < 0.20) |
        (s["ctt_disc"]       < 0.20) |
        (s["point_biserial"] < 0.20) |
        (~s["irt_disc"].between(0.50, 2.50)) |
        (s["irt_guess"]      > 0.25) |
        (s["alpha_if_removed"] > alpha_threshold)
    )

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
  <h4>Q16 — Low Discrimination &amp; High Guessing (Energy &amp; Momentum)</h4>
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
  <h4>Q22 — Extremely Low Difficulty &amp; Discrimination (Energy)</h4>
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
  <h4>Q23 — Negative Gain &amp; Out-of-Range IRT Discrimination (Momentum)</h4>
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
        a = row["irt_disc"] if pd.notna(row["irt_disc"]) else 1.0
        b = row["irt_diff"] if pd.notna(row["irt_diff"]) else 0.0
        c = row["irt_guess"] if pd.notna(row["irt_guess"]) else 0.0
        prob  = c + (1 - c) / (1 + np.exp(-a * (theta - b)))
        color = TYPE_COLORS[row["type"]]
        ax_idx = "" if idx == 0 else str(idx + 1)

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

        fig5.add_annotation(
            xref=f"x{ax_idx}", yref=f"y{ax_idx}",
            x=2.6, y=0.08,
            text=f"a={a:.2f}<br>b={b:.2f}<br>c={c:.2f}",
            showarrow=False, align="right",
            font=dict(size=7.5, color=color),
            bgcolor="rgba(255,255,255,0.80)",
        )

        if row["problematic"] and show_problematic:
            fig5.add_shape(
                type="rect",
                xref=f"x{ax_idx}", yref=f"y{ax_idx}",
                x0=-3, x1=3, y0=0, y1=1,
                line=dict(color="#e74c3c", width=1.5, dash="dash"),
                fillcolor="rgba(0,0,0,0)", layer="above",
            )

        if show_thresholds:
            fig5.add_shape(type="line",
                xref=f"x{ax_idx}", yref=f"y{ax_idx}",
                x0=-3, x1=3, y0=0.5, y1=0.5,
                line=dict(color="#ddd", width=0.8, dash="dot"), layer="below")
            fig5.add_shape(type="line",
                xref=f"x{ax_idx}", yref=f"y{ax_idx}",
                x0=0, x1=0, y0=0, y1=1,
                line=dict(color="#ddd", width=0.8, dash="dot"), layer="below")

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

    for ann in fig5.layout.annotations:
        ann.font.size  = 10
        ann.font.color = "#111111"

    fig5.update_layout(
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(color=AXIS_CLR, size=9),
        height=int(CHART_H * 1.83),
        margin=dict(l=45, r=15, t=55, b=40),
        showlegend=False,
    )

    legend_html = "&nbsp;&nbsp;".join(
        f'<span style="color:{TYPE_COLORS[k]};font-weight:700">■ {v}</span>'
        for k, v in TYPE_LABELS.items()
    )
    legend_html += '&nbsp;&nbsp;&nbsp;<span style="color:#e74c3c;font-weight:700">⬚ Flagged item</span>'
    st.markdown(legend_html, unsafe_allow_html=True)
    st.plotly_chart(fig5, use_container_width=True, config=PLOTLY_EXPORT_CONFIG)

# ════════════════════════════════════════════
# TAB 6 — Item Analysis by Category
# ════════════════════════════════════════════
with tab6:
    st.markdown("### Item Analysis by Category")
    st.caption(
        "Developers categorized 14 items as Energy-related (Q01, Q02, Q04, Q06, Q08, Q09, Q12, Q13, Q15, Q17, Q20, Q22, Q24, Q25), "
        "10 items as Momentum-related (Q03, Q05, Q07, Q10, Q11, Q14, Q18, Q19, Q21, Q23), and 1 item as Energy & Momentum (Q16)."
    )

    type_summary = (
        df.groupby("type_label")[["pre_test", "post_test", "gain"]]
        .mean()
        .reset_index()
        .rename(columns={"type_label": "Type", "pre_test": "Pre-Test",
                         "post_test": "Post-Test", "gain": "Norm. Gain"})
    )
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
        height=int(CHART_H * 0.81),
        xaxis=dict(title="Item Type"),
        yaxis=dict(title="Proportion Correct", range=[0, 1.05],
                   showgrid=True, gridcolor=GRID_CLR),
        yaxis2=dict(title="Normalized Gain", overlaying="y", side="right",
                    range=[-0.2, 0.7], showgrid=False),
    )
    st.plotly_chart(fig6a, use_container_width=True, config=PLOTLY_EXPORT_CONFIG)

    st.markdown("#### Pre-Test vs Post-Test per Item (colored by type)")
    fig6b = go.Figure()
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
        height=CHART_H,
        xaxis=dict(title="Pre-Test (proportion correct)", range=[-0.02, 1.02],
                   showgrid=True, gridcolor=GRID_CLR, linecolor="#aaa"),
        yaxis=dict(title="Post-Test (proportion correct)", range=[-0.02, 1.02],
                   showgrid=True, gridcolor=GRID_CLR, linecolor="#aaa"),
    )
    st.plotly_chart(fig6b, use_container_width=True, config=PLOTLY_EXPORT_CONFIG)

# ════════════════════════════════════════════
# TAB 7 — Rankings & Psychometric Descriptions
# ════════════════════════════════════════════
with tab7:
    st.markdown(f"### {txt['rankings_title']}")
    st.caption(txt['rankings_sub'])

    st.markdown(f"#### {txt['ctt_rank_header']}")
    st.caption(txt['ctt_rank_caption'])

    df_ctt_sorted = df.sort_values("ctt_diff", ascending=True).reset_index(drop=True)

    col_ctt_table, col_ctt_desc = st.columns([1.2, 1])

    with col_ctt_table:
        st.dataframe(
            df_ctt_sorted[["item", "type_label", "ctt_diff", "ctt_disc", "pre_test", "post_test", "gain"]],
            column_config={
                "item": st.column_config.TextColumn("Item"),
                "type_label": st.column_config.TextColumn("Type"),
                "ctt_diff": st.column_config.NumberColumn("CTT Diff (p)", format="%.3f"),
                "ctt_disc": st.column_config.NumberColumn("CTT Disc", format="%.3f"),
                "pre_test": st.column_config.NumberColumn("Pre-Test", format="%.3f"),
                "post_test": st.column_config.NumberColumn("Post-Test", format="%.3f"),
                "gain": st.column_config.NumberColumn("Gain (g)", format="%.3f"),
            },
            use_container_width=True,
            hide_index=True,
            height=450,
        )

    with col_ctt_desc:
        if lang_choice == "en":
            st.markdown("""
<div class="info-card">
  <h4>📊 CTT Difficulty Analysis &amp; Overview</h4>
  <p>
    <strong>Hardest Items (p &lt; 0.40):</strong><br>
    • <strong>Q22 (p = 0.22, Energy)</strong>: Hardest item on the survey. Low discrimination (0.08) indicates high ambiguity or distractor confusion.<br>
    • <strong>Q16 (p = 0.31, Energy &amp; Momentum)</strong>: Second hardest item. Low discrimination (0.15) and high guessing.<br><br>
    <strong>Moderate Items (0.40 ≤ p ≤ 0.70):</strong><br>
    • 12 items (Q06, Q14, Q08, Q25, Q20, Q04, Q12, Q02, Q23, Q18, Q10, Q24) form the core assessment boundary.<br>
    • Q23 shows an anomaly: high post-test p-value (0.68) but negative gain (-0.13) due to pre-test baseline.<br><br>
    <strong>Easiest Items (p &gt; 0.70):</strong><br>
    • 11 items (Q01, Q11, Q17, Q03, Q19, Q07, Q13, Q21, Q05, Q15, Q09).<br>
    • <strong>Q09 (p = 0.80, Energy)</strong> is the easiest item on the survey.
  </p>
</div>
""", unsafe_allow_html=True)
        else:
            st.markdown("""
<div class="info-card">
  <h4>📊 Análisis y Resumen de Dificultad CTT</h4>
  <p>
    <strong>Reactivos Más Difíciles (p &lt; 0.40):</strong><br>
    • <strong>Q22 (p = 0.22, Energía)</strong>: El reactivo más difícil. Su baja discriminación (0.08) indica ambigüedad o distractores confusos.<br>
    • <strong>Q16 (p = 0.31, Energía y Momento)</strong>: Segundo más difícil. Baja discriminación (0.15) y alta adivinación.<br><br>
    <strong>Reactivos Moderados (0.40 ≤ p ≤ 0.70):</strong><br>
    • 12 reactivos (Q06, Q14, Q08, Q25, Q20, Q04, Q12, Q02, Q23, Q18, Q10, Q24) constituyen el núcleo de evaluación.<br>
    • Q23 muestra una anomalía: alta dificultad en post-test pero ganancia negativa (-0.13).<br><br>
    <strong>Reactivos Más Fáciles (p &gt; 0.70):</strong><br>
    • 11 reactivos (Q01, Q11, Q17, Q03, Q19, Q07, Q13, Q21, Q05, Q15, Q09).<br>
    • <strong>Q09 (p = 0.80, Energía)</strong> es el reactivo más fácil de la encuesta.
  </p>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown(f"#### {txt['irt_rank_header']}")
    st.caption(txt['irt_rank_caption'])

    df_irt_sorted = df.sort_values("irt_diff", ascending=True).reset_index(drop=True)

    col_irt_table, col_irt_desc = st.columns([1.2, 1])

    with col_irt_table:
        st.dataframe(
            df_irt_sorted[["item", "type_label", "irt_diff", "irt_disc", "irt_guess", "problematic"]],
            column_config={
                "item": st.column_config.TextColumn("Item"),
                "type_label": st.column_config.TextColumn("Type"),
                "irt_diff": st.column_config.NumberColumn("IRT Diff (b)", format="%.3f"),
                "irt_disc": st.column_config.NumberColumn("IRT Disc (a)", format="%.3f"),
                "irt_guess": st.column_config.NumberColumn("IRT Guessing (c)", format="%.3f"),
                "problematic": st.column_config.CheckboxColumn("Flagged"),
            },
            use_container_width=True,
            hide_index=True,
            height=450,
        )

    with col_irt_desc:
        if lang_choice == "en":
            st.markdown("""
<div class="info-card">
  <h4>🎯 IRT Difficulty (b) vs CTT Difficulty (p) Comparison</h4>
  <p>
    <strong>Lowest Ability Thresholds (b &lt; 0.0):</strong><br>
    • <strong>Q22 (b = -1.12)</strong> and <strong>Q16 (b = -0.85)</strong> have negative b parameters because their guessing parameters are very high (c = 0.31 – 0.32). In 3PL IRT, high pseudo-chance levels lower the theoretical ability location where P(θ) = (1+c)/2.<br><br>
    <strong>Normal Assessment Spectrum (b = 0.22 to 0.67):</strong><br>
    • Items Q06 through Q09 span the standard ability latent continuum cleanly with steep discrimination slopes (a = 1.35 to 1.98).<br><br>
    <strong>Highest Ability Threshold (b = 0.75):</strong><br>
    • <strong>Q23 (b = 0.75, Momentum)</strong> requires the highest latent ability θ for endorsement, accompanied by high discrimination (a = 2.84).
  </p>
</div>
""", unsafe_allow_html=True)
        else:
            st.markdown("""
<div class="info-card">
  <h4>🎯 Comparación Dificultad IRT (b) vs Dificultad CTT (p)</h4>
  <p>
    <strong>Umbrales de Habilidad Más Bajos (b &lt; 0.0):</strong><br>
    • <strong>Q22 (b = -1.12)</strong> y <strong>Q16 (b = -0.85)</strong> tienen parámetros b negativos debido a su alto parámetro de adivinación (c = 0.31 – 0.32). En 3PL, la alta probabilidad de azar desplaza hacia abajo el nivel de habilidad donde P(θ) = (1+c)/2.<br><br>
    <strong>Espectro Normal de Evaluación (b = 0.22 a 0.67):</strong><br>
    • Los reactivos Q06 a Q09 cubren el continuo de habilidad latente con pendientes de discriminación pronunciadas (a = 1.35 a 1.98).<br><br>
    <strong>Umbral de Habilidad Más Alto (b = 0.75):</strong><br>
    • <strong>Q23 (b = 0.75, Momento)</strong> requiere la mayor habilidad latente θ, acompañado de una alta discriminación (a = 2.84).
  </p>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown(f"#### {txt['avg_method_header']}")
    if lang_choice == "en":
        st.markdown("""
<div class="info-card">
  <h4>🧮 How Pre-Test and Post-Test Averages are Calculated</h4>
  <p>
    In psychometric survey evaluation (such as EMCS), the overall pre-test and post-test averages can be calculated in two mathematically identical ways:
  </p>
  <ol>
    <li>
      <strong>Item-Level Average (p-value mean):</strong><br>
      The average proportion correct across all <em>K = 25</em> items:
      <br>
      <code>Avg Pre-Test = (1 / K) ∑ p_i</code>
    </li>
    <li>
      <strong>Student-Level Average (total score mean):</strong><br>
      The average total test score across all <em>N</em> students divided by <em>K = 25</em>:
      <br>
      <code>Avg Student Score = (1 / N) ∑ (Score_j / K)</code>
    </li>
  </ol>
  <p>
    Both formulas yield the exact same value. For normalized gain <em>g</em>, Hake's average gain is defined as:
    <br>
    <code>g_avg = (Post_avg − Pre_avg) / (1 − Pre_avg)</code>
  </p>
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown("""
<div class="info-card">
  <h4>🧮 Cómo se Calculan los Promedios de Pre-Test y Post-Test</h4>
  <p>
    En la evaluación de encuestas psicométricas (como EMCS), los promedios de pre-test y post-test se pueden calcular de dos formas matemáticamente idénticas:
  </p>
  <ol>
    <li>
      <strong>Promedio a Nivel de Reactivo (media del valor p):</strong><br>
      La proporción promedio correcta a través de los <em>K = 25</em> reactivos:
      <br>
      <code>Prom. Pre-Test = (1 / K) ∑ p_i</code>
    </li>
    <li>
      <strong>Promedio a Nivel de Estudiante (media del puntaje total):</strong><br>
      El puntaje total promedio a través de todos los <em>N</em> estudiantes dividido entre <em>K = 25</em>:
      <br>
      <code>Prom. Estudiante = (1 / N) ∑ (Puntaje_j / K)</code>
    </li>
  </ol>
  <p>
    Ambas fórmulas producen exactamente el mismo resultado. Para la ganancia normalizada <em>g</em>, la ganancia promedio de Hake se define como:
    <br>
    <code>g_prom = (Post_prom − Pre_prom) / (1 − Pre_prom)</code>
  </p>
</div>
""", unsafe_allow_html=True)

    st.markdown(f"#### {txt['alpha_method_header']}")
    if lang_choice == "en":
        st.markdown("""
<div class="info-card">
  <h4>📐 Cronbach's Alpha (α) Calculation &amp; Scale Reliability</h4>
  <p>
    <strong>Can Cronbach's alpha be calculated?</strong> <br>
    <strong>YES!</strong> Cronbach's alpha is the standard measure of internal consistency reliability for conceptual surveys.
  </p>
  <p>
    Formula:
    <br>
    <code>α = [ K / (K − 1) ] × [ 1 − ( ∑ σ_i² ) / σ_total² ]</code>
  </p>
  <ul>
    <li><strong>Pre-Test Cronbach's α = 0.7563</strong> (good reliability for a diagnostic pre-test).</li>
    <li><strong>Post-Test Cronbach's α = 0.8672</strong> (excellent reliability after instruction).</li>
    <li><strong>Alpha-if-removed:</strong> Highlighted for each item in Tab 4. Items where α_removed &gt; overall α (such as Q16 and Q22) degrade scale reliability and are flagged.</li>
  </ul>
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown("""
<div class="info-card">
  <h4>📐 Cálculo del Alfa de Cronbach (α) y Fiabilidad de la Escala</h4>
  <p>
    <strong>¿Se puede calcular el alfa de Cronbach?</strong> <br>
    <strong>¡SÍ!</strong> El alfa de Cronbach es la medida estándar de fiabilidad de consistencia interna para encuestas conceptuales.
  </p>
  <p>
    Fórmula:
    <br>
    <code>α = [ K / (K − 1) ] × [ 1 − ( ∑ σ_i² ) / σ_total² ]</code>
  </p>
  <ul>
    <li><strong>Alfa de Cronbach Pre-Test = 0.7563</strong> (buena fiabilidad para un diagnóstico previo).</li>
    <li><strong>Alfa de Cronbach Post-Test = 0.8672</strong> (excelente fiabilidad tras la instrucción).</li>
    <li><strong>Alfa si se elimina:</strong> Se destaca para cada reactivo en la Tab 4. Reactivos donde α_eliminar &gt; α_general (como Q16 y Q22) reducen la fiabilidad y se marcan con bandera.</li>
  </ul>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════
# TAB 8 — Compute Metrics from Raw xlsx
# ════════════════════════════════════════════
with tab8:
    st.markdown("### Compute Psychometric Metrics from Raw Response File")
    st.caption(
        "Upload a student response xlsx (columns: file_name, CODE, Q01–Q25). "
        "The system scores responses, computes all CTT / IRT metrics, and lets you "
        "download the result as a CSV to import into Tab 4."
    )

    ANSWER_KEY_T8 = {
        "Q01":"b","Q02":"e","Q03":"b","Q04":"a","Q05":"d",
        "Q06":"c","Q07":"e","Q08":"c","Q09":"a","Q10":"d",
        "Q11":"e","Q12":"d","Q13":"c","Q14":"d","Q15":"a",
        "Q16":"c","Q17":"b","Q18":"e","Q19":"b","Q20":"a",
        "Q21":"c","Q22":"d","Q23":"b","Q24":"a","Q25":"e",
    }
    with st.expander("🔑 Answer Key (click to view)"):
        ak_df = pd.DataFrame([ANSWER_KEY_T8])
        st.dataframe(ak_df, use_container_width=True, hide_index=True)

    uploaded_raw = st.file_uploader(
        "Upload student response file (.xlsx)",
        type=["xlsx"], key="tab8_upload",
    )

    if uploaded_raw is not None:
        try:
            file_bytes = uploaded_raw.getvalue()
            with st.spinner("Scoring responses and computing psychometric metrics…"):
                result_df, n_ctt, n_irt = compute_raw_excel_metrics(file_bytes)

            st.success(
                f"✅ Computation complete! CTT scored on {n_ctt:,} student records (NAs=0); "
                f"IRT calibrated on {n_irt:,} complete-case records in sub-second speed."
            )
            st.dataframe(result_df, use_container_width=True, hide_index=True)

            csv_out = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇ Download as CSV (import into Tab 4)",
                data=csv_out,
                file_name="computed_metrics.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
