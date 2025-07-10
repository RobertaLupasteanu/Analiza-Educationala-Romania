import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from data_loader import get_clean_data, get_demographics_data
from utils import COUNTY_CODE_TO_NAME, normalize_county
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.tree import export_graphviz

import graphviz
from sklearn.metrics import r2_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
import numpy as np
from utils import ordinal_encode_nivel, binary_encode_mediu

# Configurare pagină
st.set_page_config(page_title="Analiză de clasificare", page_icon="📉", layout="wide")
st.title("📉 Analiză de clasificare")

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

#region --- Regresie logistică: predicția abandonului școlar ---

st.header("📊 Regresie logistică pentru predicția abandonului școlar")

st.subheader(
    f"""
    Calcul rata de abandon școlar
    """)
# Formula afișată cu suport LaTeX
st.latex(r"\mathrm{dropout\_rate} = \frac{elevi_{2023-2024} - elevi_{2022-2023}}{elevi_{2023-2024}} \times 100")

st.markdown(
    f"""
    <div style="font-size:23px; line-height:1.4;">
    Unde:<br>
    - elevi_2023-2024: numărul de elevi înscriși în anul școlar 2023–2024<br>
    - elevi_2022-2023: numărul de elevi înscriși în anul școlar 2022–2023<br>
    Această rată exprimă procentual pierderea elevilor față de anul curent.
    </div>
    """, unsafe_allow_html=True
)

# 1) Agregăm numărul de elevi per județ și perioadă
agg_enroll = (
    df
    .groupby(['Judet','period'])['elevi_existenti']
    .sum()
    .reset_index()
)

# 2) Pivotăm și calculăm rate
pivot_enroll = agg_enroll.pivot(index='Judet', columns='period', values='elevi_existenti')
periods = sorted(pivot_enroll.columns)
if len(periods) < 2:
    st.warning("Nu sunt suficiente perioade pentru calculul ratei de abandon.")
else:
    prev, last = periods[-2], periods[-1]
    pivot_enroll['dropout_rate'] = ((pivot_enroll[last] - pivot_enroll[prev]) / pivot_enroll[last]) * 100
    if len(periods) >= 3:
        third = periods[-3]
        pivot_enroll['dropout_rate_prev'] = ((pivot_enroll[prev] - pivot_enroll[third]) / pivot_enroll[prev]) * 100
    else:
        pivot_enroll['dropout_rate_prev'] = pivot_enroll['dropout_rate']

    # 3) Îmbinăm cu schimbarea populației
demo = (
    get_demographics_data()[['Judet_code','change_2022_2023']]
    .rename(columns={'change_2022_2023':'pop_change'})
)
df_log = (
    pivot_enroll.reset_index()
    .merge(demo, left_on='Judet', right_on='Judet_code', how='left')
    .dropna(subset=['pop_change','dropout_rate'])
)

# 4) Definim ținta binară
df_log['target'] = (df_log['dropout_rate'] > df_log['dropout_rate'].mean()).astype(int)

# 5) Pregătim și antrenăm modelul
X_log = df_log[['pop_change','dropout_rate_prev']].values
y_log = df_log['target'].values
model_log = LogisticRegression().fit(X_log, y_log)
y_pred_proba = model_log.predict_proba(X_log)[:,1]
y_pred = model_log.predict(X_log)
auc = roc_auc_score(y_log, y_pred_proba)
coefs = model_log.coef_[0]
intercept = model_log.intercept_[0]

# 6) Afișăm AUC și ROC cu interpretări lângă grafic
st.subheader("🔍 Rezultate regresie logistică")
col_chart, col_interp = st.columns([2,3])
with col_chart:
    fig_log, ax_log = plt.subplots(figsize=(6,4))
    fpr, tpr, _ = roc_curve(y_log, y_pred_proba)
    ax_log.plot(fpr, tpr, linewidth=2)
    ax_log.plot([0,1],[0,1],'k--')
    ax_log.set_xlabel('False Positive Rate')
    ax_log.set_ylabel('True Positive Rate')
    ax_log.set_title('Curba ROC')
    st.pyplot(fig_log)
with col_interp:
    st.markdown(
            f"""
            <div style="font-size:23px; line-height:1.4;">
                <h4><strong>Interpretare Curba ROC</strong> </h4>
                <ul>
                <li>
                    <strong>AUC = {auc:.3f}</strong> In cazul nostru, AUC = {auc:.3f} indică că modelul
                    are o probabilitate de {auc*100:.1f}% de a clasifica corect un județ cu rată de abandon
                        ridicată față de unul cu rată scăzută.
                </li>
                <li><strong>Forma curbei</strong> curba deviată semnificativ de diagonala punctată 
                    arată că modelul diferențiază bine între cele două clase.
                </li>
                <h4><strong>Parametrii modelului</strong> </h4>
                <li>
                    <strong>Intercept = {intercept:.3f}</strong>
                    log-odds când toate feature-urile sunt zero; sugerează un punct de pornire al modelului.
                </li>
                <li><strong>Coeficient pop_change = {coefs[0]:.3f}.</strong>
                <strong> Odds ratio = </strong>
                    {np.exp(coefs[0]):.3f}, adică o unitate de creștere a `pop_change` crește odds-ul de abandon
                    cu {np.exp(coefs[0]):.3f}x.
                </li>
                <li><strong>Coeficient dropout_rate_prev = {coefs[1]:.3f}.</strong>
                <strong> Odds ratio = </strong>
                    {np.exp(coefs[1]):.3f}, indicând că o creștere cu 1 punct procentual
                    în rata anterioară de abandon multiplică odds-ul curent de abandon cu {np.exp(coefs[1]):.3f}x.
                </li>
                </ul>

            </div>
            """, unsafe_allow_html=True
        )

# 7) Matrice de confuzie și interpretări dedesubt
cm = confusion_matrix(y_log, y_pred)
cm_df = pd.DataFrame(cm, index=["Negativ real","Pozitiv real"], columns=["Negativ pred","Pozitiv pred"])
st.write(f"""<div style="font-size:23px; line-height:1.4;"><strong>Matrice de confuzie</strong></div>""", unsafe_allow_html=True)
st.dataframe(cm_df)
st.markdown(
     f"""
    <div style="font-size:20px; line-height:1.4;">
     <h4><strong>Interpretare Matrice de Confuzie: </strong></h4>
     <ul>
        <li>
            <strong>True Negative (TN): </strong> 
            numărul de județe corect identificate cu abandon scăzut (poză: {cm[0,0]}).
        </li>
        <li>
            <strong>False Positive (FP): </strong> 
            județe prezise cu abandon ridicat, dar reale cu abandon scăzut ({cm[0,1]}).
        </li>
        <li>
            <strong>False Negative (FN): </strong> 
            județe cu abandon ridicat prezise scăzut ({cm[1,0]}).
        </li>
        <li>
            <strong>True Positive (TP): </strong> 
            județe corect identificate cu abandon ridicat ({cm[1,1]}).
        </li>
     </ul>
     </div>
     """, unsafe_allow_html=True
)

# 8) Raport de clasificare
st.write(f"""<div style="font-size:23px; line-height:1.4;"><strong>Classification report</strong></div>""", unsafe_allow_html=True)
report_dict = classification_report(y_log, y_pred, output_dict=True)
df_report = pd.DataFrame(report_dict).transpose()
# Formatare numerice cu 2 zecimale
st.dataframe(df_report.style.format({col: "{:.2f}" for col in df_report.columns}))

st.markdown(
    f"""
    <div style='font-size:22px; line-height:1.5;'>
      <h4><strong>Interpretare Classification Report</strong></h4>
      <ul>
        <li><strong>Precision (clasa 0 - abandon scăzut):</strong> {df_report.loc['0','precision']:.2f}  
            <em>- {df_report.loc['0','precision']*100:.1f}% dintre predicțiile de 'scăzut' sunt corecte.</em>
        </li>
        <li><strong>Recall (clasa 0):</strong> {df_report.loc['0','recall']:.2f}  
            <em>- {df_report.loc['0','recall']*100:.1f}% dintre județele reale cu abandon scăzut au fost identificate.</em>
        </li>
        <li><strong>Precision (clasa 1 - abandon ridicat):</strong> {df_report.loc['1','precision']:.2f}  
            <em>- {df_report.loc['1','precision']*100:.1f}% dintre predicțiile de 'ridicat' sunt corecte.</em>
        </li>
        <li><strong>Recall (clasa 1):</strong> {df_report.loc['1','recall']:.2f}  
            <em>- {df_report.loc['1','recall']*100:.1f}% dintre județele reale cu abandon ridicat au fost identificate.</em>
        </li>
        <li><strong>F1-score:</strong> nivel abandon scolar - scăzut = {df_report.loc['0','f1-score']:.2f}, 
            nivel abandon scolar - ridicat = {df_report.loc['1','f1-score']:.2f}  
            <br><i>F1-score reprezintă media armonică dintre precision și recall 
            (2·precision·recall/(precision+recall)), 
            oferind o măsură echilibrată atunci când distribuția claselor este dezechilibrată.</i>
            <br>Aici F1 pentru clasa scăzută = {df_report.loc['0','f1-score']:.2f} 
            sugerează un echilibru bun între acuratețe și sensibilitate, 
            iar pentru clasa ridicată = {df_report.loc['1','f1-score']:.2f} indică cât de robust predicțiile
              de abandon ridicat sunt în contextul true positive și false negative.</br>
        </li>
        <li><strong>Support:</strong> nivel abandon scolar - scăzut = {int(df_report.loc['0','support'])}, 
            nivel abandon scolar - ridicat = {int(df_report.loc['1','support'])}  
            <em>- Numărul de observații utilizate pentru fiecare clasă (ex. combinații județ-pe-an), nu județe unice.</em>
        </li>
      </ul>
    </div>
    """, unsafe_allow_html=True
)

col_chart, col_interp = st.columns([1,1])

with col_chart:
    # 1) Date pentru curba logistică (fixăm pop_change la mediană)
    x_var = df_log['dropout_rate_prev'].values
    y_bin = df_log['target'].values
    x_grid = np.linspace(x_var.min(), x_var.max(), 200)
    pop_med = np.median(df_log['pop_change'].values)
    X_curve = np.column_stack([np.full_like(x_grid, pop_med), x_grid])
    y_proba = model_log.predict_proba(X_curve)[:,1]

    # 2) Threshold aproximativ P=0.5
    thresh_idx = np.argmin(np.abs(y_proba - 0.5))
    thresh_val = x_grid[thresh_idx]

    # 3) Jitter pe y pentru punctele binare
    y_jitter = y_bin + np.random.normal(0, 0.03, size=len(y_bin))

    # 4) Plot
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(
        x_var, y_jitter,
        c=y_bin, cmap='bwr',
        alpha=0.4, s=30, edgecolors='none'
    )
    ax.plot(x_grid, y_proba, color='green', linewidth=2)
    ax.axvline(thresh_val, linestyle='--', color='gray', alpha=0.7)
    ax.set_xlabel('Dropout Rate Prev (%)')
    ax.set_ylabel('P(abandon ridicat)')
    ax.set_title('Curba logistică vs Dropout Rate Prev')
    st.pyplot(fig)

with col_interp:
    st.markdown(
        f"""
        <div style="font-size:20px; line-height:1.4;">
          <h4><strong>Interpretarea curbei logistice</strong></h4>
          <ul>
            <li>
              <strong>Punctele albastre</strong> (0) și <strong>roșii</strong> (1) sunt observațiile reale, 
              ușor jitter‐uite pe verticală pentru claritate.
            </li>
            <li>
              <strong>Curba verde</strong> afișează probabilitatea modelului de a prezice 
              <em>abandon ridicat</em> în funcție de <code>dropout_rate_prev</code>, 
              menținând <code>pop_change</code> fix la mediana sa.
            </li>
            <li>
              Linia punctată gri marchează pragul aproximativ 
              <strong>Dropout Rate Prev = {thresh_val:.2f}%</strong> 
              unde P(abandon ridicat) trece de 0.5.
            </li>
            <li>
              Sub acest prag, modelul estimează P&lt;0.5 (favorizează <em>abandon scăzut</em>), 
              iar peste el P&gt;0.5 (favorizează <em>abandon ridicat</em>).
            </li>
            <li>
              Forma S-ului indică cum crește probabilitatea în mod rapid în jurul valorii de 
              <strong>{thresh_val:.2f}%</strong>, care este zona de maximă sensibilitate.
            </li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True
    )
#endregion


#region --- Arbore de decizie: definire, antrenare și vizualizare ---
st.header("🌳 Arbore de decizie pentru predicția abandonului școlar")

# --- Definirea modelului și a datelor ---
st.markdown(
    f"""
    <div style="font-size:20px; line-height:1.4;">
      <h4><strong>Despre model</strong></h4>
      <p>
        Folosim un <em>Decision Tree Classifier</em> pentru a prezice dacă un județ 
        va avea rată de abandon şcolar ridicată sau scăzută.
        Modelul:
      </p>
      <ul>
        <li>Primește ca input toate variabilele disponibile (pop_change, dropout_rate_prev, 
            nivel codificat, mediu, etc.).</li>
        <li>Alege secvențial feature-ul și pragul care reduc impuritatea (Gini) la fiecare nod.</li>
        <li>Constrânge adâncimea la valoarea aleasă de utilizator pentru a evita overfitting-ul.</li>
        <li>La final, fiecare frunză clasifică județele în “Abandon scăzut” sau “Abandon ridicat”.</li>
      </ul>
    </div>
    """,
    unsafe_allow_html=True
)

st.subheader("🌳 Arbore decizional pentru predicția abandonului școlar")

# 1) Slider pentru adâncimea arborelui
st.markdown(f"""
    <div style="font-size:20px; line-height:1.4;"><strong>Adâncimea maximă a arborelui</strong></div>""",
    unsafe_allow_html=True)
max_depth = st.slider( "Max depth",
    min_value=1, max_value=15,
    value=6, step=1
)

# 2) Agregăm o valoare pe județ (moda) pentru categoriile brute
cat_raw = df[['Judet','Nivel','Mediu','Profil','Tip finantare']].copy()
df_cat = (
    cat_raw
      .groupby('Judet')
      .agg(lambda s: s.mode().iat[0] if not s.mode().empty else 'Other')
      .reset_index()
)

# 3) Recodări custom
real_set = {'Real','Matematică-Real','Științe reale'}
tech_set = {'Tehnic','Tehnologic','Industrial'}
human_set = {'Umanist','Filologie','Științe sociale'}
df_cat['Profil_grp'] = df_cat['Profil'].apply(lambda p:
    'Real' if p in real_set else
    'Tehnic' if p in tech_set else
    'Umanist' if p in human_set else
    'Other'
)

age_map = {'Gimnazial':'11-14','Liceal':'14-18','Profesional':'14-18'}
df_cat['Age_grp'] = df_cat['Nivel'].map(age_map).fillna('Other')

df_cat['Mediu_grp'] = df_cat['Mediu'].apply(lambda m:
    m if m in ['Rural','Urban'] else 'Other'
)

df_cat['Tip_grp'] = df_cat['Tip finantare'].apply(lambda t:
    t if t in ['BUGET','TAXA'] else 'Other'
)

# 4) Îmbinăm cu df_log (unde avem target-ul și variabile numerice pop_change, dropout_rate_prev)
df_tree = (
    df_log
      .merge(df_cat[['Judet','Profil_grp','Age_grp','Mediu_grp','Tip_grp']],
             on='Judet', how='left')
      .dropna(subset=['Profil_grp','Age_grp','Mediu_grp','Tip_grp'])
)

# 5) Construim matricea de feature-uri:
#    - one-hot pentru categorice
#    - numeric direct pentru pop_change și dropout_rate_prev
features_cat = ['Profil_grp','Age_grp','Mediu_grp','Tip_grp']
X_cat = pd.get_dummies(df_tree[features_cat], drop_first=False)
X_num = df_tree[['pop_change','dropout_rate_prev']].reset_index(drop=True)
X = pd.concat([X_cat, X_num], axis=1)
y = df_tree['target'].values

# # 6) Afișăm coloanele rezultate
# st.subheader("📑 Feature matrix columns")
# for col in X.columns:
#     st.text(f"- {col}")

# 7) Antrenăm arborele
model = DecisionTreeClassifier(
    max_depth=max_depth,
    class_weight='balanced',
    random_state=42
).fit(X, y)

# 8) Regulile arborelui în text
st.subheader("📋 Regulile arborelui (text)")
rules = export_text(model, feature_names=list(X.columns))
st.text(rules)
# 6) Afișăm coloanele rezultate
st.subheader("📑 Feature matrix columns")
for col in X.columns:
    st.text(f"- {col}")

# # 6.1) Tabel cu intervale
# st.subheader("📊 Intervalele de valori ale variabilelor")
# df_ranges = X.agg(['min','max']).T.reset_index()
# df_ranges.columns = ['Feature','Min','Max']
# st.table(df_ranges)

# 7) Antrenăm arborele
model = DecisionTreeClassifier(
    max_depth=max_depth,
    class_weight='balanced',
    random_state=42
).fit(X, y)

# 9) Vizualizare arbore + interpretare în paralel
st.subheader("🌳 Arbore decizional")


# Folosim un canvas mai lat și mai scund, ca să-l întindem orizontal
fig, ax = plt.subplots(figsize=(20, 6))
plot_tree(
    model,
    feature_names=list(X.columns),
    class_names=['Scăzut','Ridicat'],
    filled=True, rounded=True, proportion=True,
    fontsize=8,
    ax=ax
)
ax.set_xticks([])
ax.set_ylabel("")
st.pyplot(fig)
#endregion