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

# ── Custom CSS ala dashboard pemerintah ──────────────────────
st.markdown("""
<style>
    .main { background-color: #f0f2f6; }
    .block-container { padding: 2rem 3rem; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 5px solid #1a3c6e;
    }
    .metric-label { font-size: 13px; color: #666; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-value { font-size: 32px; font-weight: 700; color: #1a3c6e; margin: 4px 0; }
    .metric-sub   { font-size: 14px; color: #888; }
    .section-header {
        background: linear-gradient(135deg, #1a3c6e, #2563a8);
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        font-size: 16px;
        font-weight: 600;
        margin: 20px 0 12px 0;
        letter-spacing: 0.3px;
    }
    .badge-sangat-bahagia  { background:#d1fae5; color:#065f46; padding:6px 16px; border-radius:20px; font-weight:700; font-size:15px; }
    .badge-bahagia         { background:#dbeafe; color:#1e40af; padding:6px 16px; border-radius:20px; font-weight:700; font-size:15px; }
    .badge-cukup-bahagia   { background:#fef9c3; color:#854d0e; padding:6px 16px; border-radius:20px; font-weight:700; font-size:15px; }
    .badge-kurang-bahagia  { background:#ffedd5; color:#9a3412; padding:6px 16px; border-radius:20px; font-weight:700; font-size:15px; }
    .badge-tidak-bahagia   { background:#fee2e2; color:#991b1b; padding:6px 16px; border-radius:20px; font-weight:700; font-size:15px; }
    h1 { color: #1a3c6e !important; }
    .stSelectbox label, .stSlider label { font-weight: 600; color: #333; }
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
    if score >= 7.0:   return "😄 Sangat Bahagia",  "sangat-bahagia"
    elif score >= 6.0: return "🙂 Bahagia",          "bahagia"
    elif score >= 5.0: return "😐 Cukup Bahagia",    "cukup-bahagia"
    elif score >= 4.0: return "😔 Kurang Bahagia",   "kurang-bahagia"
    else:              return "😢 Tidak Bahagia",     "tidak-bahagia"

feature_labels = {
    "Log GDP per capita":               "Log GDP per Capita",
    "Social support":                   "Social Support",
    "Perceptions of corruption":        "Perceptions of Corruption",
    "Positive affect":                  "Positive Affect",
    "Negative affect":                  "Negative Affect",
    "affect_balance":                   "Affect Balance",
}

# ── Header ───────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 11])
with col_title:
    st.markdown("# 🌏 Dashboard Prediksi Tingkat Kebahagiaan Negara")
    st.markdown("**World Happiness Report — Life Ladder Prediction System** | Powered by XGBoost")
st.divider()

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Flag_of_Indonesia.svg/320px-Flag_of_Indonesia.svg.png", width=80)
    st.markdown("### ⚙️ Parameter Input")
    st.markdown("---")

    countries = sorted(df["Country name"].unique())
    negara    = st.selectbox("🌍 Pilih Negara", countries, index=countries.index("Indonesia"))

    years_available = sorted(df[df["Country name"] == negara]["year"].unique())
    tahun = st.selectbox("📅 Pilih Tahun", years_available, index=len(years_available)-1)

    st.markdown("---")
    st.markdown("### 🎛️ Atau Input Manual")
    st.caption("Geser slider untuk override nilai dari dataset")

    use_manual = st.toggle("Aktifkan Input Manual", value=False)

    # Ambil data negara + tahun dari dataset
    row = df[(df["Country name"] == negara) & (df["year"] == tahun)]

    input_values = {}
    feature_ranges = {
        "Log GDP per capita":        (5.5,  12.0, 0.01),
        "Social support":            (0.0,  1.0,  0.01),
        "Perceptions of corruption": (0.0,  1.0,  0.01),
        "Positive affect":           (0.0,  1.0,  0.01),
        "Negative affect":           (0.0,  1.0,  0.01),
        "affect_balance":            (-1.0, 1.0,  0.01),
    }

    for feat in FEATURES_SELECTED:
        label        = feature_labels.get(feat, feat)
        mn, mx, step = feature_ranges.get(feat, (0.0, 1.0, 0.01))
        default_val  = float(row[feat].values[0]) if not row.empty else (mn + mx) / 2

        if use_manual:
            input_values[feat] = st.slider(label, mn, mx, default_val, step)
        else:
            input_values[feat] = default_val

    predict_btn = st.button("🔮 Prediksi Sekarang", type="primary", use_container_width=True)

# ── Main content ─────────────────────────────────────────────
if not row.empty:
    actual_score = float(row[TARGET].values[0])
    label_actual, badge_actual = interpret_happiness(actual_score)

    # ── Section 1: Info negara ───────────────────────────────
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

    # ── Section 2: Hasil prediksi (kalau tombol ditekan) ────
    if predict_btn:
        X_input  = np.array([[input_values[f] for f in FEATURES_SELECTED]])
        pred     = xgb.predict(X_input)[0]
        label_pred, badge_pred = interpret_happiness(pred)
        selisih  = pred - actual_score

        st.markdown(f'<div class="section-header">🔮 Hasil Prediksi XGBoost</div>', unsafe_allow_html=True)

        p1, p2, p3 = st.columns(3)
        with p1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Prediksi Life Ladder</div>
                <div class="metric-value">{pred:.4f}</div>
                <div class="metric-sub">Hasil model XGBoost</div>
            </div>""", unsafe_allow_html=True)
        with p2:
            arrow = "▲" if selisih >= 0 else "▼"
            warna = "#065f46" if selisih >= 0 else "#991b1b"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Selisih vs Aktual</div>
                <div class="metric-value" style="color:{warna}">{arrow} {abs(selisih):.4f}</div>
                <div class="metric-sub">Prediksi vs nilai aktual</div>
            </div>""", unsafe_allow_html=True)
        with p3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Interpretasi</div>
                <div class="metric-value" style="font-size:20px">{label_pred}</div>
                <div class="metric-sub">Kategori kebahagiaan</div>
            </div>""", unsafe_allow_html=True)

        # Tabel input yang dipakai
        st.markdown("##### 📋 Nilai Fitur yang Digunakan")
        df_input = pd.DataFrame({
            "Fitur":  [feature_labels.get(f, f) for f in FEATURES_SELECTED],
            "Nilai":  [f"{input_values[f]:.4f}" for f in FEATURES_SELECTED],
        })
        st.dataframe(df_input, use_container_width=True, hide_index=True)

    # ── Section 3: Grafik tren ───────────────────────────────
    st.markdown(f'<div class="section-header">📈 Tren Life Ladder — {negara}</div>', unsafe_allow_html=True)

    tren = df[df["Country name"] == negara].sort_values("year")

    fig, ax = plt.subplots(figsize=(13, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8fafc")

    ax.plot(tren["year"], tren[TARGET], color="#1a3c6e", linewidth=2.5,
            marker="o", markersize=6, label="Life Ladder Aktual", zorder=3)
    ax.fill_between(tren["year"], tren[TARGET], alpha=0.08, color="#1a3c6e")

    # Highlight tahun yang dipilih
    if tahun in tren["year"].values:
        hl_val = tren[tren["year"] == tahun][TARGET].values[0]
        ax.scatter([tahun], [hl_val], color="#e63946", s=120, zorder=5, label=f"Tahun dipilih ({int(tahun)})")
        ax.annotate(f"{hl_val:.2f}", (tahun, hl_val),
                    textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=10, color="#e63946", fontweight="bold")

    # Garis referensi kategori
    refs = [(7.0, "#065f46", "Sangat Bahagia"), (6.0, "#1e40af", "Bahagia"),
            (5.0, "#854d0e", "Cukup Bahagia"),  (4.0, "#9a3412", "Kurang Bahagia")]
    for val, color, lbl in refs:
        ax.axhline(val, color=color, linewidth=0.8, linestyle="--", alpha=0.4)
        ax.text(tren["year"].min() - 0.3, val + 0.05, lbl,
                fontsize=7.5, color=color, alpha=0.7, va="bottom")

    ax.set_xlabel("Tahun", fontsize=11)
    ax.set_ylabel("Life Ladder Score", fontsize=11)
    ax.set_title(f"Tren Life Ladder — {negara} ({int(tren['year'].min())}–{int(tren['year'].max())})",
                 fontsize=13, fontweight="bold", color="#1a3c6e")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    # ── Section 4: Perbandingan global ──────────────────────
    st.markdown(f'<div class="section-header">🏆 Perbandingan Global — Tahun {int(tahun)}</div>', unsafe_allow_html=True)

    global_year  = df[df["year"] == tahun].copy()
    global_rank  = global_year.sort_values(TARGET, ascending=False).reset_index(drop=True)
    global_rank.index += 1
    rank_negara  = global_rank[global_rank["Country name"] == negara].index[0]
    total_negara = len(global_rank)

    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Peringkat Global</div>
            <div class="metric-value">#{rank_negara}</div>
            <div class="metric-sub">dari {total_negara} negara ({int(tahun)})</div>
        </div>""", unsafe_allow_html=True)
    with r2:
        global_mean = global_year[TARGET].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Rata-rata Global</div>
            <div class="metric-value">{global_mean:.3f}</div>
            <div class="metric-sub">Rata-rata semua negara</div>
        </div>""", unsafe_allow_html=True)
    with r3:
        vs_global = actual_score - global_mean
        arrow = "▲" if vs_global >= 0 else "▼"
        warna = "#065f46" if vs_global >= 0 else "#991b1b"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">vs Rata-rata Global</div>
            <div class="metric-value" style="color:{warna}">{arrow} {abs(vs_global):.3f}</div>
            <div class="metric-sub">Selisih dari rata-rata dunia</div>
        </div>""", unsafe_allow_html=True)

    # Top 10 & Bottom 10
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
    st.warning(f"⚠️ Data untuk **{negara}** tahun **{int(tahun)}** tidak ditemukan di dataset.")

# ── Footer ───────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center; color:#888; font-size:12px'>
    Life Ladder Prediction Dashboard • World Happiness Report Dataset • Model: XGBoost Regressor
</div>
""", unsafe_allow_html=True)