import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Life Ladder Dashboard",
    page_icon="🌏",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────────────

st.markdown("""
<style>
    .block-container { padding: 2rem 3rem; }
    .metric-card {
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        border-left: 5px solid #4268F6;
        border: 1px solid rgba(128,128,128,0.2);
        margin-bottom: 8px;
    }

    /* Light mode */
    @media (prefers-color-scheme: light) {
        .metric-card { background: #f8fafc; }
        .metric-label { color: #555555 !important; }
        .metric-value { color: #1a3c6e !important; }
        .metric-sub   { color: #777777 !important; }
    }

    /* Dark mode */
    @media (prefers-color-scheme: dark) {
        .metric-card { background: #1e2a3a; }
        .metric-label { color: #aaaaaa !important; }
        .metric-value { color: #ffffff !important; }
        .metric-sub   { color: #cccccc !important; }
    }

    /* Fallback kalau media query tidak terbaca */
    .metric-label { font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-value { font-size: 32px; font-weight: 700; margin: 4px 0; }
    .metric-sub   { font-size: 14px; }

    .section-header {
        background: linear-gradient(135deg, #1a3c6e, #2563a8);
        color: white !important;
        padding: 12px 20px;
        border-radius: 8px;
        font-size: 16px;
        font-weight: 600;
        margin: 20px 0 12px 0;
        letter-spacing: 0.3px;
    }
</style>
""", unsafe_allow_html=True)
# ── Load model & data ────────────────────────────────────────
@st.cache_resource
def load_model():
    xgb               = joblib.load("model/xgb.pkl")
    FEATURES_SELECTED = joblib.load("model/features_selected.pkl")
    return xgb, FEATURES_SELECTED

@st.cache_data
def load_data():
    url  = 'https://drive.google.com/file/d/17tGLhrDK1QieYtc_-_f_5B28gl5TrGhm/view?usp=sharing'
    path = 'https://drive.google.com/uc?export=download&id=' + url.split('/')[-2]
    df   = pd.read_csv(path, sep=";", decimal=",", encoding="utf-8-sig")

    TARGET   = "Life Ladder"
    FEATURES = [
        "Log GDP per capita", "Social support",
        "Healthy life expectancy at birth", "Freedom to make life choices",
        "Generosity", "Perceptions of corruption",
        "Positive affect", "Negative affect",
    ]
    df = df.dropna(subset=[TARGET]).copy()
    for col in FEATURES:
        df[col] = df.groupby("Country name")[col].transform(lambda x: x.fillna(x.mean()))
        df[col] = df[col].fillna(df[col].mean())
    df["affect_balance"] = df["Positive affect"] - df["Negative affect"]
    return df

xgb, FEATURES_SELECTED = load_model()
df = load_data()
TARGET = "Life Ladder"

# ── Helper ───────────────────────────────────────────────────
def interpret_happiness(score):
    if score >= 7.0:   return "😄 Sangat Bahagia"
    elif score >= 6.0: return "🙂 Bahagia"
    elif score >= 5.0: return "😐 Cukup Bahagia"
    elif score >= 4.0: return "😔 Kurang Bahagia"
    else:              return "😢 Tidak Bahagia"

feature_labels = {
    "Log GDP per capita":               "Log GDP per Capita",
    "Social support":                   "Social Support",
    "Healthy life expectancy at birth": "Healthy Life Expectancy",
    "Perceptions of corruption":        "Perceptions of Corruption",
    "Positive affect":                  "Positive Affect",
    "affect_balance":                   "Affect Balance",
}

feature_ranges = {
    "Log GDP per capita":               (5.5,  12.0, 0.01),
    "Social support":                   (0.0,  1.0,  0.01),
    "Healthy life expectancy at birth": (30.0, 80.0, 0.1),
    "Perceptions of corruption":        (0.0,  1.0,  0.01),
    "Positive affect":                  (0.0,  1.0,  0.01),
    "affect_balance":                   (-1.0, 1.0,  0.01),
}

# ── Header ───────────────────────────────────────────────────
st.markdown("# 🌏 Dashboard Prediksi Tingkat Kebahagiaan Negara")
st.markdown("**World Happiness Report — Life Ladder Prediction System** | Powered by XGBoost")
st.divider()

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Parameter Input")
    st.markdown("---")

    countries = sorted(df["Country name"].unique())
    negara    = st.selectbox("🌍 Pilih Negara", countries, index=countries.index("Indonesia"))

    years_available = sorted(df[df["Country name"] == negara]["year"].unique())
    tahun_options   = years_available + [2024, 2025, 2026]
    tahun           = st.selectbox("📅 Pilih Tahun", tahun_options, index=len(tahun_options)-1)

    row = df[(df["Country name"] == negara) & (df["year"] == tahun)]

    st.markdown("---")
    st.markdown("### 🎛️ Input Manual")
    st.caption("Aktifkan untuk override nilai fitur dari dataset")

    use_manual = st.toggle("Aktifkan Input Manual", value=row.empty)

    if row.empty:
        st.warning("⚠️ Tahun ini tidak ada di dataset. Isi nilai fitur secara manual.")

    input_values = {}
    for feat in FEATURES_SELECTED:
        label        = feature_labels.get(feat, feat)
        mn, mx, step = feature_ranges.get(feat, (0.0, 1.0, 0.01))
        default_val  = float(row[feat].values[0]) if not row.empty else (mn + mx) / 2

        if use_manual:
            input_values[feat] = st.slider(label, mn, mx, default_val, step)
        else:
            input_values[feat] = default_val

    predict_btn = st.button("🔮 Prediksi Sekarang", type="primary", use_container_width=True)

# ── Section 1: Info negara (hanya kalau ada di dataset) ──────
if not row.empty:
    actual_score = float(row[TARGET].values[0])
    label_actual = interpret_happiness(actual_score)

    st.markdown(f'<div class="section-header">📌 Informasi Negara — {negara} ({int(tahun)})</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Life Ladder Aktual</div>
            <div class="metric-value">{actual_score:.3f}</div>
            <div class="metric-sub">Skor kebahagiaan aktual</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        gdp_val = float(row["Log GDP per capita"].values[0])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Log GDP per Capita</div>
            <div class="metric-value">{gdp_val:.3f}</div>
            <div class="metric-sub">Indikator ekonomi</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        ss_val = float(row["Social support"].values[0])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Social Support</div>
            <div class="metric-value">{ss_val:.3f}</div>
            <div class="metric-sub">Dukungan sosial</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Status Kebahagiaan</div>
            <div class="metric-value" style="font-size:20px">{label_actual}</div>
            <div class="metric-sub">Berdasarkan skor aktual</div>
        </div>""", unsafe_allow_html=True)
else:
    st.info(f"ℹ️ Tahun **{int(tahun)}** tidak ada di dataset — prediksi berdasarkan input manual")

# ── Section 2: Hasil prediksi ────────────────────────────────
if predict_btn:
    if row.empty and not use_manual:
        st.warning("⚠️ Aktifkan **Input Manual** di sidebar untuk mengisi nilai fitur!")
    else:
        X_input    = np.array([[input_values[f] for f in FEATURES_SELECTED]])
        pred       = xgb.predict(X_input)[0]
        label_pred = interpret_happiness(pred)

        st.markdown(f'<div class="section-header">🔮 Hasil Prediksi XGBoost — {negara} ({int(tahun)})</div>', unsafe_allow_html=True)

        p1, p2, p3 = st.columns(3)
        with p1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Prediksi Life Ladder</div>
                <div class="metric-value">{pred:.4f}</div>
                <div class="metric-sub">Hasil model XGBoost</div>
            </div>""", unsafe_allow_html=True)
        with p2:
            if not row.empty:
                selisih = pred - float(row[TARGET].values[0])
                arrow   = "▲" if selisih >= 0 else "▼"
                warna   = "#4ade80" if selisih >= 0 else "#f87171"
                sub     = "Prediksi vs nilai aktual"
                val_str = f"{arrow} {abs(selisih):.4f}"
            else:
                warna   = "#888888"
                sub     = "Tidak ada data aktual"
                val_str = "—"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Selisih vs Aktual</div>
                <div class="metric-value" style="color:{warna}">{val_str}</div>
                <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)
        with p3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Interpretasi</div>
                <div class="metric-value" style="font-size:20px">{label_pred}</div>
                <div class="metric-sub">Kategori kebahagiaan</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("##### 📋 Nilai Fitur yang Digunakan")
        df_input = pd.DataFrame({
            "Fitur": [feature_labels.get(f, f) for f in FEATURES_SELECTED],
            "Nilai": [f"{input_values[f]:.4f}" for f in FEATURES_SELECTED],
        })
        st.dataframe(df_input, use_container_width=True, hide_index=True)

# ── Section 3: Grafik tren (hanya kalau ada di dataset) ──────
if not row.empty or tahun in df[df["Country name"] == negara]["year"].values:
    tren = df[df["Country name"] == negara].sort_values("year")

    if len(tren) > 0:
        st.markdown(f'<div class="section-header">📈 Tren Life Ladder — {negara}</div>', unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(13, 4.5))
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#1e2a3a")

        ax.plot(tren["year"], tren[TARGET], color="#4268F6", linewidth=2.5,
                marker="o", markersize=6, label="Life Ladder Aktual", zorder=3)
        ax.fill_between(tren["year"], tren[TARGET], alpha=0.1, color="#4268F6")

        if tahun in tren["year"].values:
            hl_val = tren[tren["year"] == tahun][TARGET].values[0]
            ax.scatter([tahun], [hl_val], color="#f76a8c", s=120, zorder=5, label=f"Tahun dipilih ({int(tahun)})")
            ax.annotate(f"{hl_val:.2f}", (tahun, hl_val),
                        textcoords="offset points", xytext=(0, 12),
                        ha="center", fontsize=10, color="#f76a8c", fontweight="bold")

        refs = [(7.0, "#4ade80", "Sangat Bahagia"), (6.0, "#60a5fa", "Bahagia"),
                (5.0, "#fbbf24", "Cukup Bahagia"),  (4.0, "#f87171", "Kurang Bahagia")]
        for val, color, lbl in refs:
            ax.axhline(val, color=color, linewidth=0.8, linestyle="--", alpha=0.5)
            ax.text(tren["year"].min() - 0.3, val + 0.05, lbl,
                    fontsize=7.5, color=color, alpha=0.8, va="bottom")

        ax.set_xlabel("Tahun", fontsize=11, color="#cccccc")
        ax.set_ylabel("Life Ladder Score", fontsize=11, color="#cccccc")
        ax.set_title(f"Tren Life Ladder — {negara} ({int(tren['year'].min())}–{int(tren['year'].max())})",
                     fontsize=13, fontweight="bold", color="#ffffff")
        ax.tick_params(colors="#aaaaaa")
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.spines[:].set_color("#2e3a4e")
        ax.legend(fontsize=10, labelcolor="white", facecolor="#1e2a3a", edgecolor="#2e3a4e")
        ax.grid(True, alpha=0.2, color="#ffffff")
        plt.tight_layout()
        st.pyplot(fig)

# ── Section 4: Perbandingan global (hanya kalau tahun ada) ───
global_year_df = df[df["year"] == tahun]
if not global_year_df.empty:
    st.markdown(f'<div class="section-header">🏆 Perbandingan Global — Tahun {int(tahun)}</div>', unsafe_allow_html=True)

    global_rank  = global_year_df.sort_values(TARGET, ascending=False).reset_index(drop=True)
    global_rank.index += 1
    total_negara = len(global_rank)
    global_mean  = global_year_df[TARGET].mean()

    negara_in_rank = global_rank[global_rank["Country name"] == negara]

    r1, r2, r3 = st.columns(3)
    with r1:
        if not negara_in_rank.empty:
            rank_negara = negara_in_rank.index[0]
            rank_str    = f"#{rank_negara}"
            rank_sub    = f"dari {total_negara} negara ({int(tahun)})"
        else:
            rank_str = "—"
            rank_sub = "Tidak ada di dataset tahun ini"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Peringkat Global</div>
            <div class="metric-value">{rank_str}</div>
            <div class="metric-sub">{rank_sub}</div>
        </div>""", unsafe_allow_html=True)
    with r2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Rata-rata Global</div>
            <div class="metric-value">{global_mean:.3f}</div>
            <div class="metric-sub">Rata-rata semua negara</div>
        </div>""", unsafe_allow_html=True)
    with r3:
        if not row.empty:
            vs_global = float(row[TARGET].values[0]) - global_mean
            arrow     = "▲" if vs_global >= 0 else "▼"
            warna     = "#4ade80" if vs_global >= 0 else "#f87171"
            val_str   = f"{arrow} {abs(vs_global):.3f}"
            sub       = "Selisih dari rata-rata dunia"
        else:
            warna   = "#888888"
            val_str = "—"
            sub     = "Tidak ada data aktual"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">vs Rata-rata Global</div>
            <div class="metric-value" style="color:{warna}">{val_str}</div>
            <div class="metric-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**🌟 Top 10 Negara Paling Bahagia**")
        top10 = global_rank.head(10)[["Country name", TARGET]].copy()
        top10.columns = ["Negara", "Life Ladder"]
        top10["Life Ladder"] = top10["Life Ladder"].round(3)
        top10.index = range(1, 11)
        st.dataframe(top10, use_container_width=True)
    with t2:
        st.markdown("**😔 Bottom 10 Negara Kurang Bahagia**")
        bot10 = global_rank.tail(10)[["Country name", TARGET]].copy()
        bot10.columns = ["Negara", "Life Ladder"]
        bot10["Life Ladder"] = bot10["Life Ladder"].round(3)
        bot10.index = range(total_negara - 9, total_negara + 1)
        st.dataframe(bot10, use_container_width=True)
else:
    st.info(f"ℹ️ Data perbandingan global untuk tahun {int(tahun)} tidak tersedia di dataset")

# ── Footer ───────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center; color:#888; font-size:12px'>
    Life Ladder Prediction Dashboard • World Happiness Report Dataset • Model: XGBoost Regressor
</div>
""", unsafe_allow_html=True)
