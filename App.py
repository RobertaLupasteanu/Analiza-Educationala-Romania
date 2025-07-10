import streamlit as st
from data_loader import get_clean_data

# Configure page
st.set_page_config(
    page_title="Analiză Educațională România",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("Analiză Educațională România")

# Data overview paragraph
st.markdown(
    f"""
    <div style="font-size:20px; line-height:1.4;">
      <h4><strong>Despre setul de date</strong></h4>
      <p>
        Acest set de date conține informații privind <strong>numărul de elevi</strong>, <strong>ratele de abandon școlar</strong>, 
        precum și date demografice și geografice pentru fiecare județ din România, acoperind perioadele școlare 
        de la <strong>2019-2020</strong> până în <strong>2023-2024</strong>.
      </p>
      <p>
        Scopul analizei este de a oferi o perspectivă <strong>exploratorie</strong>, <strong>predictivă</strong> și 
        <strong>statistică</strong> asupra factorilor care influențează performanța și abandonul școlar, utilizând multiple 
        tehnici de modelare și vizualizare.
      </p>
      <p>
        Aplicația este structurată în module:
        <ul>
          <li><strong>Explorare</strong> – vizualizări descriptive și hărți.</li>
          <li><strong>Predictivă</strong> – analiză PCA și previziuni.</li>
          <li><strong>Clusterizare</strong> – identificarea grupărilor de județe.</li>
          <li><strong>Regresie</strong> – modele de regresie multiplă.</li>
          <li><strong>Clasificare</strong> – clasificarea elevilor în categorii de risc.</li>
        </ul>
      </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Navigation overview with summaries
st.header("Navigare Aplicație")
st.markdown(
    "Folosește butoanele de mai jos pentru a accesa diferitele secțiuni și vezi un scurt rezumat al fiecăreia:"
)
nav_cols = st.columns(5)
pages = [
    ("Explorare", "pages/1_analiza_exploratorie.py", "Statistici descriptive și vizualizări exploratorii."),
    ("Predictivă", "pages/2_analiza_predictiva.py", "Analiză PCA și modele predictive."),
    ("Clusterizare", "pages/3_analiza_cluster.py", "Determinarea grupărilor de județe."),
    ("Regresie", "pages/4_analiza_regresiei.py", "Model de regresie multiplă."),
    ("Clasificare", "pages/5_analiza_de_clasificare.py", "Clasificarea elevilor în categorii de risc.")
]
for col, (label, target, desc) in zip(nav_cols, pages):
    with col:
        if st.button(label):
            st.switch_page(target)
        st.caption(desc)

# Load & prepare data (cached)
@st.cache_data(show_spinner=True)
def get_data():
    # Încarcă și curăță datele inițial
    return get_clean_data()

data = get_data()

with st.expander("Previzualizare Date Importate & Sumar (" + f"{data.shape[0]:,} rânduri, {data.shape[1]} coloane" + ")", expanded=False):
    st.dataframe(data.head(), use_container_width=True)
    st.markdown(f"**Rânduri:** {data.shape[0]:,} &nbsp;|&nbsp; **Coloane:** {data.shape[1]}")


# KPI metrics
st.header("Indicatori Cheie")
col1, col2 = st.columns(2)
col1.metric("Ani analizați", f"{data['period'].nunique()}")
data = data.dropna(subset=['Judet'])
num_judete = data['Judet'].nunique() - 1
col2.metric("Județe incluse", f"{num_judete} (municipiul București este inclus)")
# Afișare lista județe
judete_list = sorted(data['Judet'].unique())
st.markdown(
    "**Listă județe incluse:** " + 
    ", ".join(judete_list) 
)
# col3.metric(
#     "Total elevi înscriși (2019-2024)",
#     f"{int(data['elevi_existenti'].sum()):,}"
# )

# Footer
st.markdown("---")
st.caption("© 2025 Analiza predictiva pentru un set mare de date - Lupasteanu Andreea-Roberta")
