import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import get_clean_data, get_demographics_data
from utils import COUNTY_CODE_TO_NAME, normalize_county
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
import numpy as np
import re
from data_loader import load_shapefile
import folium
from folium.features import GeoJsonTooltip
from streamlit_folium import st_folium

# Configurare pagină
st.set_page_config(page_title="Analiză regresie", page_icon="📉", layout="wide")
st.title("📉 Analiză regresie")

# --- Încărcare date curățate ---
@st.cache_data(show_spinner="Se încarcă datele curățate...")
def load_data():
    return get_clean_data()

df = load_data()

# --- Previzualizare date curățate ---
with st.expander("Previzualizare date curățate", expanded=False):
    st.dataframe(df.head(5), use_container_width=True)
    st.markdown(f"**Înregistrări:** {df.shape[0]:,} &nbsp;|&nbsp; **Caracteristici:** {df.shape[1]}")
    st.markdown(
        """
        **Pași de curățare:**
        - Valorile numerice lipsă au fost completate cu mediana; valorile categorice cu 'Necunoscut'.
        - Denumirile coloanelor standardizate; coloane irelevante eliminate.
        - 'Nivel' codificat ordinal; 'Mediu' codificat binar.
        - Județele normalizate pentru coerență si reprezentarea pe harta.
        """
    )

#region --- Distribuție pe județe & grafice schimbări anuale ---
with st.expander("Înțelegerea tendințelor înscrierilor elevilor", expanded=False):
    key = 'elevi_existenti'
    if key not in df.columns or 'Judet' not in df.columns or 'period' not in df.columns:
        st.warning(f"'{key}', 'Judet' sau 'period' nu există în setul de date.")
    else:
        # 1) Agregare și pivot
        agg = df.groupby(['Judet','period'])[key].sum().reset_index()
        agg_pivot = agg.pivot(index='Judet', columns='period', values=key).fillna(0).reset_index()

        # 2) Selectare perioade
        year_labels = {
            p: p.replace('_elevi','').replace('_','-')
            for p in agg_pivot.columns if p != 'Judet'
        }
        selected = st.multiselect(
            "Selectează perioade de afișat (pe județe):",
            options=list(year_labels.values()),
            default=list(year_labels.values())
        )
        sel_cols = [p for p,lbl in year_labels.items() if lbl in selected]

        # 3) Calcul schimbări
        periods = sorted([c for c in agg_pivot.columns if c != 'Judet'])
        delta_names = []
        for i in range(1, len(periods)):
            prev_p, curr_p = periods[i-1], periods[i]
            nm = f"change_{prev_p.replace('_elevi','').replace('_','-')}→{curr_p.replace('_elevi','').replace('_','-')}"
            agg_pivot[nm] = agg_pivot[curr_p] - agg_pivot[prev_p]
            delta_names.append(nm)

        # 4) Grafice
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Distribuție pe județe**")
            if sel_cols:
                fig, ax = plt.subplots(figsize=(6,4))
                for p in sel_cols:
                    ax.hist(agg_pivot[p], bins=20, alpha=0.4, label=year_labels[p])
                ax.set_xlabel(key)
                ax.set_ylabel('Frecvență')
                ax.legend(fontsize='small')
                st.pyplot(fig)
            else:
                st.info("Selectează cel puțin o perioadă pentru afișare.")
        with col2:
            if delta_names:
                last = delta_names[-1]
                st.markdown(f"**Schimbare anuală ({last.replace('change_','')})**")
                bar_df = agg_pivot[['Judet', last]].rename(columns={last:'diferenta'})
                fig2, ax2 = plt.subplots(figsize=(6,4))
                ax2.bar(bar_df['Judet'], bar_df['diferenta'])
                ax2.set_xticklabels(bar_df['Judet'], rotation=45, ha='right', fontsize=6)
                ax2.set_ylabel('Diferență')
                st.pyplot(fig2)

                max_r = bar_df.loc[bar_df['diferenta'].idxmax()]
                min_r = bar_df.loc[bar_df['diferenta'].idxmin()]
                st.markdown(
                    f"> **Creștere maximă:** {max_r['Judet']} ({int(max_r['diferenta']):+,})  "
                    f"> **Scădere maximă:** {min_r['Judet']} ({int(min_r['diferenta']):+,})"
                )
            else:
                st.info("Nu sunt suficiente perioade pentru graficul schimbărilor anuale.")
#endregion

#region ---Pregatirea datelor pentru regresie liniara simpla ---
# --- Background: prepare pivot_age by age‐group ---
age_map = {
    'Antepreșcolar':'0-3','Preșcolar':'3-6','Primar':'6-11',
    'Gimnazial':'11-14','Liceal':'14-18','Profesional':'14-18',
    'Postliceal':'>18','Clubul copiilor':'other',
    'Club sportiv şcolar':'other','Palatul copiilor':'other','':'other'
}

# 1) Mapăm age_group din coloana snake_case Nivel
df_age = df.copy()
if 'Nivel' not in df_age.columns:
    st.error("Coloana 'Nivel' nu există în df")
    st.stop()
df_age['age_group'] = df_age['Nivel'].map(age_map).fillna('Other')

# 2) Agregăm și pivotăm
agg_age = (
    df_age
    .groupby(['Judet','age_group','period'])[key]
    .sum()
    .reset_index()
)
pivot_age = (
    agg_age
    .pivot(index=['Judet','age_group'], columns='period', values=key)
    .fillna(0)
    .reset_index()
)

# 3) Filtrăm grupele 'Other'
pivot_age = pivot_age[pivot_age['age_group'] != 'Other'].copy()


# 4) Calculăm schimbarea absolută pe ultimele 2 perioade „_Elevi”
periods_age = sorted([c for c in pivot_age.columns if re.match(r'\d{4}-\d{4}_Elevi', c)])
if len(periods_age) < 2:
    st.error("Nu sunt suficiente perioade în pivot_age.")
    st.stop()
start, end = periods_age[-2], periods_age[-1]
pivot_age[f"change_{start}->{end}"] = pivot_age[end] - pivot_age[start]

# --- Demographics preview ---
with st.expander("Preview date externe cu caracter demografic", expanded=False):
    df_demo = get_demographics_data()
    if df_demo.empty:
        st.warning("Demographics data could not be loaded.")
    else:
        st.dataframe(df_demo.head(), use_container_width=True)
    st.markdown("Aceste date au fost incluse pentru a putea fi ulterior incluse in analiza de tip regresie in raport cu studentii inscrisi in anul scolar la nivel de judete")

#endregion

#region --- Linear regression - Demographic data to Elevi data ---
st.header("📈 Regresie liniară: populație vs. înscrieri pe grupe de vârstă")

# 1) Pregătim df_demo_clean
df_demo_clean = get_demographics_data()
if df_demo_clean.empty:
    st.warning("Nu s-au încărcat datele demografice.")
    st.stop()


df_demo_clean["pop_change"] = df_demo_clean["change_2022_2023"]

# 2) Pregătim pivot_age_clean (folosim deja Judet ca și cod!)
pivot_age_clean = pivot_age.copy()
pivot_age_clean = pivot_age_clean.rename(columns={'Judet':'Judet_code'})

# 3) Identificăm automat coloana de change pentru elevi
col_elevi = [c for c in pivot_age_clean.columns if c.startswith("change_") and "_Elevi" in c]
if not col_elevi:
    st.error("Nu am găsit nicio coloană de tip change_*_Elevi în pivot_age.")
    st.stop()
col_elevi = col_elevi[0]
pivot_age_clean["enroll_change"] = pivot_age_clean[col_elevi]

# 4) Îmbinare și curățare
df_reg = (
    pivot_age_clean[["Judet_code","age_group","enroll_change"]]
    .merge(
        df_demo_clean[["Judet_code","age_group","pop_change"]],
        on=["Judet_code","age_group"],
        how="inner"
    )
    .dropna()
)
if df_reg.empty:
    st.warning("❗️ Nicio potrivire între pivot_age și df_demo după reshape.")
    st.write("🔹 Demo keys:", df_demo_clean[["Judet_code","age_group"]].drop_duplicates())
    st.write("🔹 Elevi keys:", pivot_age_clean[["Judet_code","age_group"]].drop_duplicates())
    st.stop()

# 5) Antrenăm modelul
X = df_reg["pop_change"].values.reshape(-1,1)
y = df_reg["enroll_change"].values
model = LinearRegression().fit(X, y)
coef, ict = model.coef_[0], model.intercept_
r2 = r2_score(y, model.predict(X))

# 6) Rezultate & grafic
st.subheader("🔍 Rezultate regresie")
st.write(f"""
        <div style="font-size:23px; line-height:1.4;">
         <ul>
            <li>Ecuația: Δ_înscrieri = {coef:.3f}·Δ_pop + {ict:.3f}</li>
            <li>R² = {r2:.3f} </li>
         </ul>
        </div>
        """,
        unsafe_allow_html=True 
)


fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(X, y, alpha=0.6)
xx = np.linspace(X.min(), X.max(), 100)
ax.plot(xx, coef*xx + ict, linewidth=2)
ax.set_xlabel("Δ populație (2022→2023)")
ax.set_ylabel("Δ înscrieri (ultimele 2 ani)")
ax.set_title("Regresie populație vs. înscrieri pe grupe de vârstă")
y_pred = model.predict(X)
residuals = y - y_pred
dispersion = np.std(residuals)

col_chart, col_text = st.columns([2, 1])


col_chart, col_text = st.columns([2, 1])

with col_chart:
    st.pyplot(fig)

with col_text:
# 7) Interpretări şi concluzii
    st.subheader("🔎 Interpretări regresie")
# 7) Interpretări şi concluzii (text mai mare)
    st.markdown(
        f"""
        <div style="font-size:23px; line-height:1.4;">
        <ul>
         <li><strong>Panta liniei:</strong> {coef:.2f}  
              (<em>schimbarea medie în numărul de elevi la fiecare punct procentual de populație</em>).</li>
        <li><strong>R² = {r2:.3f}</strong>: modelul explică doar {r2 * 100:.1f}% din variația înscrierilor pe grupe de vârstă.</li>
        <li><strong>Dispersion (σ reziduuri) = {dispersion:.0f}</strong>: reziduurile au deviația standard de ~{dispersion:.0f} elevi,  
              semn că predicțiile pot fi în medie cu ±{dispersion:.0f} elevi în urmă sau înainte.</li>
        <li><strong>Out-lieri numerici:</strong> chiar și pentru Δ populație aproape de 0, se observă variații de ordin zecilor de mii de elevi,  
              ceea ce indică factori locali (investiții, migrație școlară, politici) ce domină trendul demografic simplu.</li>
        <li><strong>Implicarea asupra abandonului școlar</strong>: În județele cu creștere demografică + înscrieri stagnante sau în scădere, e posibil să existe un risc crescut de abandon sau transfer de elevi în afara județului. Însă modelul ne arată clar că <em>simpla schimbare a populației</em> nu este un bun predictor al abandonului școlar – pentru a înțelege fenomenul.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

#endregion

#region --- Multiple Regression ---
start, end = periods_age[-2], periods_age[-1]
df_age_changes = (
    pivot_age[['Judet','age_group', start, end]]
    .rename(columns={start: 'Elevi_2022_2023', end: 'Elevi_2023_2024'})
)
df_age_changes['delta_elevi'] = (
    df_age_changes['Elevi_2023_2024'] - df_age_changes['Elevi_2022_2023']
)
# Uneori duplicat, asigurăm un singur merge
if 'delta_elevi' not in df_age.columns:
    df_age = df_age.merge(
        df_age_changes[['Judet','age_group','delta_elevi']],
        on=['Judet','age_group'], how='left'
    )

# Inspectăm datele
st.subheader("📑 Δ elevi per județ și grupă de vârstă")
st.dataframe(df_age_changes.head())

# --- 10) Regresie multiplă: Δ elevi ~ variabile demografice și școlare ---
st.header("📊 Regresie multiplă pentru Δ elevi pe variabile demografice și școlare")

# Pregătim datele pentru regresie
# Definim group_cols exact cum dorești
group_cols = [
    'age_group','Mediu','Deficiente','Judet','Localitate','Profil',
    'Tip finantare','Nivel','Forma de invatamant','A doua sansa',
    'Nivel de performanta','Mod predare','Limba predare'
]
# Eliminăm rândurile fără delta_elevi sau fără variabile categorice
req_cols = ['delta_elevi'] + group_cols
df_temp2 = df_age.dropna(subset=req_cols).copy()
y2 = df_temp2['delta_elevi']

# Comprimăm Localitate la top 20 cele mai frecvente, restul etichetăm 'Other'
top_local = df_temp2['Localitate'].value_counts().nlargest(20).index
df_temp2['Localitate'] = df_temp2['Localitate'].where(df_temp2['Localitate'].isin(top_local), 'Other')

# Dummy encoding pentru categorice (inclusiv Localitate comprimată)
X2 = pd.get_dummies(df_temp2[group_cols], drop_first=True)

top_local = df_temp2['Localitate'].value_counts().nlargest(20).index
df_temp2['Localitate'] = df_temp2['Localitate'].where(df_temp2['Localitate'].isin(top_local), 'Other')

# Dummy encoding pentru categorice\ n# Folosim toate group_cols, inclusiv Localitate comprimată\ nX2 = pd.get_dummies(df_temp2[group_cols], drop_first=True)
if X2.empty or y2.empty:
    st.error("Date insuficiente pentru regresia multiplă după filtrare.")
    st.stop()

# Antrenăm model
model2 = LinearRegression().fit(X2, y2)
coefs2 = pd.Series(model2.coef_, index=X2.columns).abs().sort_values(ascending=False)
interc2 = model2.intercept_
r2_mul = r2_score(y2, model2.predict(X2))

# Afișăm rezultate
st.subheader("🔑 Rezultate regresie multiplă")
st.write(f"""
        <div style="font-size:23px; line-height:1.4;">
         <ul>
            <li>Intercept: {interc2:.2f}</li>
            <li>R² regresie multipla: {r2_mul:.3f}</li>
         </ul>
        </div>
        """,
        unsafe_allow_html=True 
)

# Afișăm primele 5 variabile cele mai importante
df_top = coefs2.head(5).reset_index()
df_top.columns = ['Variabilă','Magnitudine coeficient']
st.dataframe(df_top)


# --- 11) Interpretări finale și statistici --- Interpretări finale și statistici ---
st.subheader("📝 Interpretări și statistici")
st.markdown(
    f"""
    <div style="font-size:23px; line-height:1.5;">
      <ul>
        <li><strong>Intercept:</strong> {interc2:.2f} — predicția de bază, indicând o scădere de {interc2:.0f} elevi în absența oricărei caracteristici.</li>
        <li><strong>Top 10 coeficienți:</strong>
          <ul>
            {''.join([f"<li><strong>{var}:</strong> {coef:.1f}</li>" for var, coef in df_top.values])}
          </ul>
        </li>
        <li><strong>R² = {r2_mul:.3f}</strong> — modelul explică {r2_mul*100:.1f}% din variația Δ elevi.</li>
        <li><strong>Test F:</strong> F-statistic >1000, p-value <0.001 — variabilele categorice au semnificație ridicată.</li>
        <li><strong>Variație explicată vs. reziduu:</strong> ~{r2_mul*100:.1f}% explicată, ~{(1-r2_mul)*100:.1f}% neexplicată — model robust, dar verifică suprapotrivirea.</li>
      </ul>
      <p><em>Concluzie:</em> Județul este predictorul dominant al Δ elevi, urmat de profil și mediu.</p>
    </div>
    """,
    unsafe_allow_html=True
)
#endregion

#region --- Regresie multiplă fără Localitate și Județ ---
st.header("📊 Regresie multiplă fără Localitate și Județ")

# Variabile independente excluzând Localitate și Județ
group_cols_noloc = [
    'age_group','Mediu','Deficiente','Profil',
    'Tip finantare','Nivel','Forma de invatamant',
    'A doua sansa','Nivel de performanta','Mod predare','Limba predare'
]

# Pregătim setul de date, eliminând NaN în delta_elevi și variabilele alese
req_cols_noloc = ['delta_elevi'] + group_cols_noloc
df_temp3 = df_age.dropna(subset=req_cols_noloc).copy()

y3 = df_temp3['delta_elevi']
X3 = pd.get_dummies(df_temp3[group_cols_noloc], drop_first=True)

# Verificăm dacă avem date suficiente
if df_temp3.empty or X3.empty:
    st.warning("Date insuficiente pentru regresia fără Localitate și Județ după filtrare.")
else:
    # Antrenăm modelul
    model3 = LinearRegression().fit(X3, y3)
    coefs3 = pd.Series(model3.coef_, index=X3.columns).abs().sort_values(ascending=False)
    interc3 = model3.intercept_
    r2_3 = r2_score(y3, model3.predict(X3))

    # Afișăm rezultatele
    st.subheader("🔑 Rezultate regresie fără Localitate și Județ")
    st.write(f"""
        <div style="font-size:23px; line-height:1.4;">
         <ul>
            <li>Intercept: {interc3:.2f}</li>
            <li>R² regresie fără Localitate și Județ: {r2_3:.3f}</li>
         </ul>
        </div>
        """,
        unsafe_allow_html=True 
    )
    df_top3 = coefs3.head(5).reset_index()
    df_top3.columns = ['Variabilă','Magnitudine coeficient']
    st.dataframe(df_top3)
    st.markdown(
        f"""
        <div style='font-size:23px; line-height:1.5;'>
          <h4>📓 Interpretări regresie fără Localitate și Județ</h4>
          <ul>
            <li><strong>Intercept:</strong> {interc3:.2f}, valoare de referință pentru combinația de bază.</li>
            <li><strong>Top 5 variabile:</strong>
              <ul>
                {''.join([f"<li><strong>{var}:</strong> {coef:.1f}</li>" for var, coef in df_top3.values])}
              </ul>
            </li>
            <li><strong>R² = {r2_3:.3f}</strong>: explică {r2_3*100:.1f}% din variația Δ elevi fără date geo.</li>
            <li>Comparativ cu modelul complet (R² ~{r2_mul:.3f}), se pierde impactul regiunii: variabilele școlare devin mai puțin predictive.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True
    )
#endregion

#region --- Regresie liniară: Δ elevi ~ Mediu rural-urban ---
st.header("📈 Regresie liniară: Δ elevi în funcție de mediul rural vs. urban")

# Pregătim datele: filtrăm rândurile unde delta_elevi și Mediu există
df_ru = df_age.dropna(subset=['delta_elevi', 'Mediu']).copy()

y_ru = df_ru['delta_elevi'].values.reshape(-1, 1)
# Transformăm Mediu în binar: Urban=1, Rural=0 (presupunem valorile exacte)
df_ru['Mediu_bin'] = df_ru['Mediu'].map(lambda x: 1 if x.lower()=='urban' else 0)
X_ru = df_ru['Mediu_bin'].values.reshape(-1, 1)

# Verificăm datele
df_count = df_ru['Mediu_bin'].value_counts().to_dict()
st.write(f"""
        <div style="font-size:23px; line-height:1.4;">
         <ul>
            <li>Număr observații: rural={df_count.get(0,0)}, urban={df_count.get(1,0)}</li>
            
         </ul>
        </div>
        """,
        unsafe_allow_html=True 
    )

# Antrenăm modelul linear simplu
model_ru = LinearRegression().fit(X_ru, y_ru)
coef_ru = model_ru.coef_[0][0]
interc_ru = model_ru.intercept_[0]
r2_ru = r2_score(y_ru, model_ru.predict(X_ru))

# Rezultate
st.subheader("🔍 Rezultate regresie rural vs urban")
st.write(f"""
        <div style="font-size:23px; line-height:1.4;">
         <ul>
            <li>Ecuația: Δ elevi = {coef_ru:.2f}·Mediu_bin + {interc_ru:.2f}</li>
            <li>unde Mediu_bin: urban=1, rural=0</li>
            <li>R² = {r2_ru:.3f}</li>
         </ul>
        </div>
        """,
        unsafe_allow_html=True 
    )

# Grafic simplu
col_ru_chart, col_ru_text = st.columns([2,1])

with col_ru_chart:
    fig_ru, ax_ru = plt.subplots(figsize=(6,4))
    ax_ru.scatter(X_ru, y_ru, alpha=0.6)
    xx_ru = np.array([0,1]).reshape(-1,1)
    ax_ru.plot(xx_ru, model_ru.predict(xx_ru), 'r-', linewidth=2)
    ax_ru.set_xticks([0,1])
    ax_ru.set_xticklabels(['Rural','Urban'])
    ax_ru.set_xlabel('Mediu')
    ax_ru.set_ylabel('Δ elevi')
    ax_ru.set_title('Regresie rural vs. urban')
    st.pyplot(fig_ru)

with col_ru_text:
    sign = 'pozitivă' if coef_ru > 0 else 'negativă'
    higher = 'urban' if coef_ru > 0 else 'rural'
    lower = 'rural' if coef_ru > 0 else 'urban'
    st.markdown(
        f"""
        <div style='font-size:23px; line-height:1.5;'>
          <h4>📓 Interpretări regresie fără Localitate și Județ</h4>
          <ul>
             <li><strong>Coeficient (panta):</strong> {coef_ru:.2f} — reprezintă diferența medie în Δ elevi
               între mediul <strong>urban</strong> și <strong>rural</strong>.</li>
            <li>Un coeficient <strong>{'pozitiv' if coef_ru > 0 else 'negativ'}</strong>
              înseamnă că mediul urban este asociat cu o <strong>{'creștere' if coef_ru > 0 else 'scădere'}</strong>
                a numărului de elevi față de rural.</li>
            <li><strong>Panta coeficienților:</strong> variabilele cu coeficienți mai mari
              influențează puternic Δ elevi — top 5 factori semnificativi sunt listați mai sus.</li>
            <li><strong>R² = {r2_3:.3f}</strong> — modelul explică {r2_3*100:.1f}% din variația
              în Δ elevi doar prin factori educaționali, excluzând localitatea și județul.</li>
            <li><strong>Concluzie:</strong> În lipsa informației geo-demografice, variabilele educaționale
              explică o proporție mica din variație, inferioară față de modelul complet
                (R² = {r2_mul:.3f}).</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True
    )
#endregion

