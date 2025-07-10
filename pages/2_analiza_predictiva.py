import streamlit as st
from data_loader import get_clean_data, get_model_data

st.set_page_config(page_title="Analiză predictivă", layout="wide")
st.markdown("## 🔮 Analiză predictivă")
st.write(f"""<div style="font-size:23px; line-height:1.4;">Alege o analiză de machine learning: </div>""", unsafe_allow_html=True)

options = {
    "Nesupervizata": "3_analiza_cluster",
    "Supervizata - Regresie": "4_analiza_regresiei",
    "Supervizata - Clasificare": "5_analiza_de_clasificare"
}

# Afișează ca butoane pe dashboard:
cols = st.columns(len(options))
for i, (label, page) in enumerate(options.items()):
    with cols[i]:
        if st.button(label, key=label):
            st.switch_page(f"pages\{page}.py")  # Navighează către subpagină

# Încarcă datele curățate din analiza exploratorie
df = get_clean_data()
st.write(f"""<div style="font-size:23px; line-height:1.4;">Previzualizare date (din Analiza exploratorie):</div>""", unsafe_allow_html=True)
st.dataframe(df.head())
