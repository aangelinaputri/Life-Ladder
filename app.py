import streamlit as st
import numpy as np
import pandas as pd
import joblib

xgb               = joblib.load("model/xgb.pkl")
FEATURES_SELECTED = joblib.load("model/features_selected.pkl")

def interpret_happiness(score):
    if score >= 7.0:   return "😄 Sangat Bahagia"
    elif score >= 6.0: return "🙂 Bahagia"
    elif score >= 5.0: return "😐 Cukup Bahagia"
    elif score >= 4.0: return "😔 Kurang Bahagia"
    else:              return "😢 Tidak Bahagia"

st.set_page_config(page_title="Life Ladder Predictor", page_icon="🪜", layout="wide")
st.title("🪜 Life Ladder Predictor")
st.markdown("Prediksi tingkat kebahagiaan negara berdasarkan World Happiness Report")
st.divider()

feature_labels = {
    "Log GDP per capita":               "Log GDP per Capita",
    "Social support":                   "Social Support",
    "Healthy life expectancy at birth": "Healthy Life Expectancy",
    "Freedom to make life choices":     "Freedom to Make Life Choices",
    "Generosity":                       "Generosity",
    "Perceptions of corruption":        "Perceptions of Corruption",
    "Positive affect":                  "Positive Affect",
    "Negative affect":                  "Negative Affect",
    "affect_balance":                   "Affect Balance",
}

feature_ranges = {
    "Log GDP per capita":               (5.5,  12.0, 9.0,  0.01),
    "Social support":                   (0.0,  1.0,  0.8,  0.01),
    "Healthy life expectancy at birth": (30.0, 80.0, 60.0, 0.1),
    "Freedom to make life choices":     (0.0,  1.0,  0.7,  0.01),
    "Generosity":                       (-0.5, 0.7,  0.0,  0.01),
    "Perceptions of corruption":        (0.0,  1.0,  0.7,  0.01),
    "Positive affect":                  (0.0,  1.0,  0.7,  0.01),
    "Negative affect":                  (0.0,  1.0,  0.3,  0.01),
    "affect_balance":                   (-1.0, 1.0,  0.4,  0.01),
}

st.sidebar.header("📝 Input Fitur")
input_values = {}
for feat in FEATURES_SELECTED:
    label = feature_labels.get(feat, feat)
    mn, mx, default, step = feature_ranges.get(feat, (0.0, 1.0, 0.5, 0.01))
    input_values[feat] = st.sidebar.slider(label, mn, mx, default, step)

if st.sidebar.button("Prediksi", type="primary", use_container_width=True):
    X_input = np.array([[input_values[f] for f in FEATURES_SELECTED]])
    pred    = xgb.predict(X_input)[0]
    label   = interpret_happiness(pred)

    st.subheader("🔮 Hasil Prediksi")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Life Ladder Score", f"{pred:.4f}")
    with col2:
        st.metric("Interpretasi", label)

    st.divider()
    st.subheader("📝 Detail Input")
    df_input = pd.DataFrame({
        "Fitur": [feature_labels.get(f, f) for f in FEATURES_SELECTED],
        "Nilai": [f"{input_values[f]:.3f}" for f in FEATURES_SELECTED],
    })
    st.dataframe(df_input, use_container_width=True, hide_index=True)