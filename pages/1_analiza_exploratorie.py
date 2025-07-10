import streamlit as st
from data_loader import load_raw_data
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
import folium
from streamlit_folium import folium_static
import unidecode
from folium.features import GeoJsonTooltip
from sklearn.impute import SimpleImputer

st.set_page_config(layout="wide")

#region Components style
def apply_css():
    st.markdown(
        """
        <style>
        .center-text { text-align: center !important; color: #001f3f; }
        </style>
        """,
        unsafe_allow_html=True
    )

def centered_table_with_selectbox(table, select_label, select_options, select_key, width=500, height=350):
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.dataframe(table, use_container_width=False, width=width, height=height)
        st.markdown(f"<h5 class='center-text'>{select_label}</h5>", unsafe_allow_html=True)
        return st.selectbox("", options=[""] + select_options, key=select_key)

def centered_table(df, width=500, height=150):
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.dataframe(df, use_container_width=False, width=width, height=height)

PERIOD_LABELS = {
    '2019_2020_elevi': 'anul școlar 2019-2020',
    '2020-2021_elevi': 'anul școlar 2020-2021',
    '2022-2023_elevi': 'anul școlar 2022-2023',
    '2023-2024_elevi': 'anul școlar 2023-2024'
}

#endregion

#region Main function
def main(df=None):
    apply_css()

    #region --- Header and Filters ---
    header_col, year_col = st.columns([6, 2])
    with header_col:
        st.markdown("## 🔍 Analiză exploratorie", unsafe_allow_html=True)
    with year_col:
        if df is None:
            df = load_raw_data()
        raw_periods = sorted(df['period'].unique())
        friendly_periods = [PERIOD_LABELS.get(p, p) for p in raw_periods]
        friendly_to_raw = {PERIOD_LABELS.get(p, p): p for p in raw_periods}
        selected_friendly = st.selectbox(
            "",
            options=["Toți"] + friendly_periods,
            key='period_filter'
        )
        st.markdown(
            f"<div style='text-align:right; font-size:1rem; color:#666;'>"
            f"{selected_friendly if selected_friendly!='Toți' else 'Toți anii școlari'}"  
            f"</div>",
            unsafe_allow_html=True
        )

    if selected_friendly == "Toți":
        df_sel = df.copy()
        title = "Toți anii școlari"
    else:
        raw_sel = friendly_to_raw[selected_friendly]
        df_sel = df[df['period'] == raw_sel]
        title = selected_friendly
    #endregion

    #region --- Descriptive Summary ---
    st.markdown(f"<h3 class='center-text'>Analiza descriptivă pentru {title}</h3>", unsafe_allow_html=True)
    rows, cols = df_sel.shape
    st.markdown(
        f"<p class='center-text'>Rânduri: <strong>{rows}</strong> &nbsp;&nbsp; Coloane: <strong>{cols}</strong></p>",
        unsafe_allow_html=True
    )

    numeric_cols = df_sel.select_dtypes(include='number').columns.tolist()
    cat_cols = [c for c in df_sel.select_dtypes(include=['object', 'category']).columns if c != 'period']

    summary_col1, summary_col2 = st.columns([1, 1])
    table_width = 500
    table_height = 350

    with summary_col1:
        st.markdown("<h4 class='center-text'>Sumar coloane numerice</h4>", unsafe_allow_html=True)
        st.markdown(f"<p class='center-text'>Număr: <strong>{len(numeric_cols)}</strong></p>", unsafe_allow_html=True)
        if numeric_cols:
            desc = df_sel[numeric_cols].describe().T
            if '50%' in desc.columns:
                desc['median'] = desc['50%']
            sel_num = centered_table_with_selectbox(
                desc,
                "Alege coloane numerice pentru statistici detaliate",
                numeric_cols,
                select_key='num_detail_sel',
                width=table_width,
                height=table_height
            )
            if sel_num:
                col_desc = df_sel[sel_num].describe().to_frame().T
                if '50%' in col_desc.columns:
                    col_desc['median'] = col_desc['50%']
                centered_table(col_desc, width=table_width)
        else:
            st.info("Nu există coloane numerice.")

    with summary_col2:
        st.markdown("<h4 class='center-text'>Sumar coloane categorice</h4>", unsafe_allow_html=True)
        st.markdown(f"<p class='center-text'>Număr: <strong>{len(cat_cols)}</strong></p>", unsafe_allow_html=True)
        if cat_cols:
            summary = []
            for col in cat_cols:
                unique = df_sel[col].nunique(dropna=False)
                non_null = df_sel[col].dropna()
                mode_series = non_null.mode() if not non_null.empty else []
                top = mode_series.iloc[0] if hasattr(mode_series, 'iloc') and not mode_series.empty else None
                freq = df_sel[col].value_counts(dropna=False).iloc[0] if not df_sel[col].dropna().empty else 0
                summary.append({'column': col, 'unique_values': unique, 'top_value': top, 'freq': freq})
            summary_df = pd.DataFrame(summary).set_index('column')
            sel_cat = centered_table_with_selectbox(
                summary_df,
                "Alege coloane categorice pentru numărătoarea valorilor",
                cat_cols,
                select_key='cat_detail_sel',
                width=table_width,
                height=table_height
            )
            if sel_cat:
                vc = df_sel[sel_cat].value_counts(dropna=False).rename_axis(sel_cat).reset_index(name='count')
                centered_table(vc, width=table_width)
        else:
            st.info("Nu există coloane categorice.")
    #endregion

    #region --- Data Consistency ---
    st.markdown("---")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.markdown("<h4 style='text-align:left;'>Verificări consistență date</h4>", unsafe_allow_html=True)
    with c2:
        show_dupes = st.toggle("Afișează rânduri duplicate", key="show_dupes")
    with c3:
        show_mixed = st.toggle("Afișează tipuri de date mixte", key="show_mixed_types")

    if 'df_clean' not in st.session_state or st.session_state['df_clean'] is None:
        df_clean = df_sel.reset_index(drop=True).copy()
        df_clean['row_id'] = df_clean.index
        st.session_state['df_clean'] = df_clean
    else:
        df_clean = st.session_state['df_clean']

    num_duplicates = df_clean.duplicated().sum()
    if num_duplicates > 0:
        st.warning(f"⚠️ Au fost identificate {num_duplicates} rânduri duplicate în datele tale. Poți să le revizuiești sau să le elimini.")
        if show_dupes:
            dupes = df_clean[df_clean.duplicated()]
            st.dataframe(dupes.head(100))
            st.caption(f"Afișare primele 100 din {len(dupes)} rânduri duplicate.")
            csv = dupes.to_csv(index=False).encode()
            st.download_button("Descarcă toate rândurile duplicate ca CSV", csv, "duplicate_rows.csv", "text/csv")
        if st.button("Șterge rândurile duplicate (păstrează primul)"):
            df_clean = df_clean.drop_duplicates(keep='first').reset_index(drop=True)
            df_clean['row_id'] = df_clean.index
            st.session_state['df_clean'] = df_clean
            st.success(
                "Toate rândurile duplicate au fost șterse și s-a generat o nouă coloană 'row_id'. Analiza de mai jos folosește acum acest set de date curățat!"
            )
    else:
        st.success("Nu au fost găsite rânduri duplicate.")

    object_cols = df_clean.select_dtypes(include='object').columns
    mixed_type_cols = [col for col in object_cols if df_clean[col].dropna().map(type).nunique() > 1]
    if mixed_type_cols:
        st.warning(f"⚠️ Coloane cu tipuri de date mixte: {', '.join(mixed_type_cols)}")
        if show_mixed:
            for col in mixed_type_cols:
                st.write(f"Valori eșantion din coloana '{col}':")
                st.write(df_clean[col].dropna().sample(min(10, df_clean[col].dropna().shape[0])))
    else:
        st.success("Nu există coloane cu tipuri de date mixte.")

    expected_medii = {'Urban', 'Rural'}
    if 'Mediu' in df_clean.columns:
        raw_vals = set(df_clean['Mediu'].dropna().unique())
        unexpected = {v for v in raw_vals if str(v).strip().lower() not in ['nan', '']}
        unexpected = unexpected - expected_medii
        if unexpected:
            st.warning(f"⚠️ Valori neașteptate în 'Mediu': {', '.join(map(str, unexpected))}")
        else:
            st.info("Coloana 'Mediu' conține doar valori așteptate.")
    #endregion

    #region --- Data Quality ---
    st.markdown("---")
    st.markdown("<h3 class='center-text'>❗ Calitatea datelor / Analiză valori lipsă</h3>", unsafe_allow_html=True)

    threshold = st.slider(
        "Selectează pragul procentual de valori lipsă:",
        min_value=0, max_value=100, value=50,
        help="Afișează doar coloanele cu procentul de valori lipsă ≥ prag"
    )

    miss_counts = df_sel.isna().sum()
    miss_perc = (miss_counts / len(df_sel) * 100).round(2)
    quality_df = pd.DataFrame({
        'missing_count': miss_counts,
        'missing_percent': miss_perc
    })
    quality_df = quality_df[quality_df['missing_percent'] >= threshold]
    quality_df = quality_df[quality_df['missing_count'] > 0].sort_values('missing_percent', ascending=False)

    table_col, chart_col = st.columns([3, 5])

    with table_col:
        if not quality_df.empty:
            qdf_display = quality_df.copy()
            qdf_display['missing_percent'] = qdf_display['missing_percent'].astype(str) + '%'
            st.dataframe(qdf_display, use_container_width=True)
        else:
            st.info("Nu există valori lipsă peste pragul selectat.")

    with chart_col:
        if not quality_df.empty:
            chart_data = quality_df.reset_index().rename(columns={'index': 'column'})
            base = alt.Chart(chart_data).mark_bar().encode(
                x=alt.X('column:N', sort='-y', title='Coloană'),
                y=alt.Y('missing_percent:Q', title='Procent valori lipsă')
            )
            rule = alt.Chart(pd.DataFrame({'threshold': [threshold]})).mark_rule(color='red', opacity=0.5).encode(
                y='threshold:Q'
            )
            st.altair_chart((base + rule).properties(width=350), use_container_width=True)

    st.markdown("---")

    cols_to_drop = quality_df.index.tolist()
    if cols_to_drop:
        df_after_drop = df_sel.drop(columns=cols_to_drop)
        st.warning(f"Coloane eliminate (>={threshold}% valori lipsă): {', '.join(cols_to_drop)}")
        st.markdown("#### Previzualizare date după eliminarea coloanelor:")
        centered_table(df_after_drop.head(), width=900)
    else:
        df_after_drop = df_sel
        st.info("Nu există coloane de eliminat pe baza pragului de valori lipsă.")

    st.markdown("#### Completare valori numerice lipsă")
    impute_option = st.selectbox(
        "Strategie pentru tratarea valorilor lipsa din coloanele numerice:",
        ["Niciuna","Medie","Mediana","Cel mai frecvent"], key='impute'
    )
    if impute_option != "Niciuna":
        strategy_map = {"Medie":"mean","Mediana":"median","Cel mai frecvent":"most_frequent"}
        imp = SimpleImputer(strategy=strategy_map[impute_option])
        num_cols_clean = df_clean.select_dtypes(include='number').columns
        df_clean[num_cols_clean] = imp.fit_transform(df_clean[num_cols_clean])
        st.session_state['df_clean']=df_clean
        st.success(f"Valorile numerice lipsă au fost completate folosind strategia {impute_option}.")

    st.session_state['df_clean'] = df_clean.copy()
    #endregion

    #region --- Data Standardization ---
    st.markdown("---")
    c1, c2 = st.columns([2, 2])
    with c1:
        st.markdown("<h4 style='text-align:left;'>Standardizare date</h4>", unsafe_allow_html=True)
    with c2:
        show_standardize = st.toggle("Afișează previzualizare", key="show_standardize", value=False)

    st.caption("Toate datele au fost standardizate. Poți restaura coloanele de text originale folosind butonul de mai jos.")

    cat_cols_to_standardize = df_sel.select_dtypes(include=['object', 'category']).columns.tolist()

    if 'standardized_applied' not in st.session_state:
        for col in cat_cols_to_standardize:
            df_clean[col] = df_clean[col].astype(str).str.strip().str.title()
        st.session_state['df_clean'] = df_clean
        st.session_state['standardized_applied'] = True

    if st.button("Păstrează originalul (fără standardizare)"):
        for col in cat_cols_to_standardize:
            df_clean[col] = df_sel[col]
        st.session_state['df_clean'] = df_clean
        st.session_state['standardized_applied'] = False
        st.info("Valorile originale au fost restaurate pentru toate coloanele text.")

    selected_std_col = None
    if show_standardize and cat_cols_to_standardize:
        selected_std_col = st.selectbox("Selectează coloana categorică pentru previzualizare", options=cat_cols_to_standardize, key="standardize_col_sel")
        if selected_std_col:
            sample_vals = df_sel[selected_std_col].dropna().astype(str).sample(min(10, df_sel[selected_std_col].dropna().shape[0])).unique()
            standardized = [s.strip().title() for s in sample_vals]
            sample_df = pd.DataFrame({"original": sample_vals, "standardized": standardized})
            st.dataframe(sample_df)
    #endregion

    #region --- Outlier Detection Section ---
    st.markdown("---")
    numeric_cols = df_clean.select_dtypes(include='number').columns.tolist()

    def get_outlier_info(col):
        q1 = df_clean[col].quantile(0.25)
        q3 = df_clean[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        num_low = (df_clean[col] < lower).sum()
        num_high = (df_clean[col] > upper).sum()
        total = df_clean[col].dropna().shape[0]
        percent = 100 * (num_low + num_high) / total if total else 0
        return {
            "lower_bound": round(lower, 2),
            "upper_bound": round(upper, 2),
            "outliers_below": num_low,
            "outliers_above": num_high,
            "total_outliers": num_low + num_high,
            "percent_outliers": percent,
            "total_values": total
        }

    outlier_stats = []
    for col in numeric_cols:
        stats = get_outlier_info(col)
        if stats["total_outliers"] > 0:
            stats["column"] = col
            outlier_stats.append(stats)

    c1, c2, c3 = st.columns([2, 1, 0.75])
    with c1:
        st.markdown("<h4 style='text-align:left;'>Detectare outlieri</h4>", unsafe_allow_html=True)
    with c2:
        show_outliers = st.toggle("Afișează outlieri", key="show_outliers", value=True)
    with c3:
        if outlier_stats:
            csv = pd.DataFrame(outlier_stats).to_csv(index=False).encode()
            st.download_button("Descarcă sumar outlieri ca CSV", csv, "outlier_summary.csv", "text/csv")

    if show_outliers and outlier_stats:
        outlier_df = pd.DataFrame(outlier_stats)
        outlier_df = outlier_df.sort_values("percent_outliers", ascending=False)
        cols_order = ['column', 'lower_bound', 'upper_bound', 'outliers_below', 'outliers_above', 'total_outliers', 'percent_outliers', 'total_values']
        outlier_df = outlier_df[cols_order]
        styled = outlier_df.style.bar(
            subset=["percent_outliers"], color='#ffa07a'
        ).background_gradient(
            subset=["outliers_below", "outliers_above", "total_outliers"], cmap="YlOrRd"
        ).format({
            "percent_outliers": "{:.1f}%",
            "lower_bound": "{:.2f}",
            "upper_bound": "{:.2f}"
        })
        st.dataframe(styled, use_container_width=True, height=min(420, 40 * (len(outlier_df) + 1)))
        st.caption("Sumar: Toate coloanele cu outlieri detectati, sortate după procentajul de outlieri.")
    elif show_outliers:
        st.info("Nu au fost detectate outlieri în coloane numerice.")
    #endregion

    #region --- Correlation Analysis ---
    st.markdown("---")
    st.markdown("<h4 class='center-text'>Analiză corelație</h4>", unsafe_allow_html=True)

    numeric_cols = df_sel.select_dtypes(include='number').columns.tolist()

    def find_first_meaningful_col(df, numeric_cols):
        for col in numeric_cols:
            if col.lower() == "_id":
                continue
            if df[col].nunique() == df.shape[0]:
                continue
            return col
        return numeric_cols[0] if numeric_cols else None

    if not numeric_cols or len(numeric_cols) < 2:
        st.info("Nu sunt suficiente coloane numerice pentru analiza corelației.")
    else:
        st.markdown("**Harta termica a corelațiilor complete (coloane numerice)**")
        corr = df_sel[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=(3.1, 1.6))
        sns.heatmap(
            corr, annot=False, fmt=".2f", cmap='coolwarm', center=0, ax=ax, cbar=False, square=False
        )
        plt.xticks(rotation=45, ha='right', fontsize=5)
        plt.yticks(rotation=0, fontsize=5)
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("Compact: corelații puternice sunt aproape de 1 sau -1.")

        default_var = find_first_meaningful_col(df_sel, numeric_cols)
        target_col = st.selectbox(
            "Selectează variabilă:",
            options=numeric_cols,
            index=numeric_cols.index(default_var) if default_var in numeric_cols else 0,
            key="corr_target"
        )

        if target_col:
            corr_series = corr[target_col].drop(target_col).sort_values(ascending=False)
            centered_table(
                corr_series.to_frame(name='Corelație').style.format("{:.2f}"),
                width=300,
                height=120
            )
            fig2, ax2 = plt.subplots(figsize=(3.1, 1.6))
            corr_series.plot(kind='bar', ax=ax2)
            ax2.set_ylabel('Corelație', fontsize=5)
            ax2.set_xlabel('')
            ax2.set_title('', fontsize=6)
            ax2.tick_params(axis='x', labelrotation=30, labelsize=4)
            ax2.tick_params(axis='y', labelsize=4)
            plt.tight_layout()
            st.pyplot(fig2)
            top_corr = corr_series[abs(corr_series) > 0.5]
            interp = []
            if not top_corr.empty:
                pos_vars = [v for v in top_corr.index if corr_series[v] > 0.5]
                neg_vars = [v for v in top_corr.index if corr_series[v] < -0.5]
                if pos_vars:
                    interp.append(f"Corelație pozitivă puternică: {', '.join(pos_vars)}")
                if neg_vars:
                    interp.append(f"Corelație negativă puternică: {', '.join(neg_vars)}")
                st.caption(" | ".join(interp))
            else:
                st.caption("Nici o variabilă nu prezintă corelație puternică (>|0.5|) cu variabila selectată.")

    #endregion

    #region --- Correlation & Evolution for 'Elevi exist anterior-asoc' ---
    st.markdown("---")
    st.markdown("<h4 class='center-text'>Analiză corelație și evoluție pentru 'Elevi exist anterior-asoc' (Toți anii)</h4>", unsafe_allow_html=True)

    if 'Elevi exist anterior-asoc' not in df_sel.columns or 'Judet' not in df_sel.columns or 'period' not in df_sel.columns:
        st.warning("Nu s-au găsit coloanele 'Elevi exist anterior-asoc', 'Judet' sau 'period' în setul de date filtrat.")
    else:
        pivot_df = df_sel.pivot_table(
            index='Judet',
            columns='period',
            values='Elevi exist anterior-asoc',
            aggfunc='sum'
        ).fillna(0).astype(int)

        year_labels = {
            '2019_2020_elevi': '2019-2020',
            '2020-2021_elevi': '2020-2021',
            '2022-2023_elevi': '2022-2023',
            '2023-2024_elevi': '2023-2024'
        }
        color_palette = {
            '2019_2020_elevi': '#4B8BBE',
            '2020-2021_elevi': '#FFD43B',
            '2022-2023_elevi': '#306998',
            '2023-2024_elevi': '#FF6F00'
        }
        available_years = [y for y in ['2019_2020_elevi', '2020-2021_elevi', '2022-2023_elevi', '2023-2024_elevi'] if y in pivot_df.columns]
        year_options = [year_labels[y] for y in available_years]

        st.markdown("**Distribuția 'Elevi exist anterior-asoc' pe județe (Ani selectați)**")
        selected_years_friendly = st.multiselect(
            "Selectează anii de afișat:",
            options=year_options,
            default=year_options,
            key="dist_years_sel"
        )
        selected_years = [k for k,v in year_labels.items() if v in selected_years_friendly and k in pivot_df.columns]

        if not selected_years:
            st.info("Te rog selectează cel puțin un an pentru a vizualiza.")
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            any_data = False
            for year in selected_years:
                label = year_labels.get(year, year)
                data = pivot_df[year]
                if data.sum() > 0:
                    sns.histplot(
                        data, bins=30, kde=True,
                        label=label,
                        ax=ax, stat='frequency', alpha=0.45, linewidth=0
                    )
                    any_data = True
            ax.set_xlabel('Număr de elevi (Elevi exist anterior-asoc)')
            ax.set_ylabel('Frecvență')
            ax.set_title('Distribuția "Elevi exist anterior-asoc" pe județe (Ani selectați)')
            if any_data:
                ax.legend(title='An')
                st.pyplot(fig)
            else:
                st.info("Nu există date nenule pentru anii selectați.")

        st.markdown("**Schimbări anuale în 'Elevi exist anterior-asoc' pe județe**")

        year_pairs = []
        for i in range(len(available_years)-1):
            year_pairs.append((available_years[i], available_years[i+1]))
        for i in range(len(available_years)):
            for j in range(i+1, len(available_years)):
                if (available_years[i], available_years[j]) not in year_pairs:
                    year_pairs.append((available_years[i], available_years[j]))
        pair_labels = [f"{year_labels[a]} → {year_labels[b]}" for (a, b) in year_pairs]
        pair_labels.insert(0, "Afișează toate modificările anuale")

        selected_pair = st.selectbox("Selectează diferența de ani de afișat:", options=pair_labels, key="diff_years_sel")

        changes = pivot_df.copy()
        delta_cols = []
        for i in range(1, len(pivot_df.columns)):
            prev_col = pivot_df.columns[i-1]
            curr_col = pivot_df.columns[i]
            delta_col = f"Δ_{year_labels.get(prev_col, prev_col)}→{year_labels.get(curr_col, curr_col)}"
            changes[delta_col] = pivot_df[curr_col] - pivot_df[prev_col]
            delta_cols.append(delta_col)
        for i in range(len(pivot_df.columns)):
            for j in range(i+1, len(pivot_df.columns)):
                prev_col = pivot_df.columns[i]
                curr_col = pivot_df.columns[j]
                delta_col = f"Δ_{year_labels.get(prev_col, prev_col)}→{year_labels.get(curr_col, curr_col)}"
                if delta_col not in changes.columns:
                    changes[delta_col] = pivot_df[curr_col] - pivot_df[prev_col]
                    delta_cols.append(delta_col)

        changes.columns = [year_labels.get(c, c) if c in year_labels else c for c in changes.columns]

        if selected_pair == "Afișează toate modificările anuale":
            st.dataframe(
                changes[[c for c in changes.columns if c in year_labels.values() or c.startswith('Δ_')]].style.format("{:,}"),
                use_container_width=True,
                height=min(370, 30 + 22 * len(changes))
            )
            last_delta = delta_cols[-1]
        else:
            idx = pair_labels.index(selected_pair)-1
            pair = year_pairs[idx]
            delta_name = f"Δ_{year_labels[pair[0]]}→{year_labels[pair[1]]}"
            df_delta = changes[[year_labels[pair[0]], year_labels[pair[1]], delta_name]].copy()
            st.dataframe(df_delta.style.format("{:,}"), use_container_width=True, height=min(370, 30 + 22 * len(df_delta)))
            last_delta = delta_name

        st.markdown(f"**Schimbarea numărului de elevi per județ ({last_delta.replace('Δ_', '')})**")
        barplot_df = changes[[last_delta]].reset_index()
        barplot_df = barplot_df.rename(columns={last_delta: 'Difference'})
        fig_bar, ax_bar = plt.subplots(figsize=(14, 7))
        sns.barplot(data=barplot_df, x='Judet', y='Difference', ax=ax_bar)
        ax_bar.set_title(f"Schimbarea numărului de elevi per județ {last_delta.replace('Δ_', '').replace('→', '→')}")
        ax_bar.set_xlabel("Județ")
        ax_bar.set_ylabel("Diferență numărului de elevi")
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig_bar)

        st.subheader("Interpretare rezultate")
        int_years = last_delta.replace('Δ_', '').split('→')
        st.write(f"""
Graficul de mai sus arată schimbarea numărului de elevi ('Elevi exist anterior-asoc') între anii școlari {int_years[0].strip()} și {int_years[1].strip()} pentru fiecare județ.
- **Valori pozitive** indică o creștere a numărului de elevi în județ.
- **Valori negative** indică o scădere a numărului de elevi.
Acest lucru îți permite să vezi care județe au avut creștere a înscrierilor și care au înregistrat o scădere.
""")
        max_judet = barplot_df.set_index('Judet')['Difference'].idxmax()
        min_judet = barplot_df.set_index('Judet')['Difference'].idxmin()
        max_val = barplot_df.set_index('Judet')['Difference'].max()
        min_val = barplot_df.set_index('Judet')['Difference'].min()
        st.markdown(
            f"""
<div style="padding:0.7em;background:rgba(220,236,255,0.85);border-left:6px solid #3875c7;font-size:1.08em;">
<strong>Creștere cea mai mare</strong>: <span style="color:#206a00;font-weight:bold">{max_judet}</span>
 (<span style=\"color:#206a00;font-weight:bold\">{max_val:+,}</span>) &nbsp; | &nbsp;
<strong>Scădere cea mai mare</strong>: <span style="color:#c70000;font-weight:bold">{min_judet}</span>
 (<span style=\"color:#c70000;font-weight:bold\">{min_val:+,}</span>).<br>
<em>Valorile pozitive</em> indică creștere; <em>valorile negative</em> indică scădere.
</div>
""", unsafe_allow_html=True
        )

    #endregion

    #region --- Dropout Change Map ---
    st.markdown("## 🗺️ Hartă: Schimbarea abandonului școlar pe județe")

    COUNTY_CODE_TO_NAME = {
        "AB": "ALBA", "AR": "ARAD", "AG": "ARGES", "BC": "BACAU", "BH": "BIHOR", "BN": "BISTRITA NASAUD",
        "BR": "BRAILA", "BT": "BOTOSANI", "BV": "BRASOV", "BZ": "BUZAU", "CS": "CARAS SEVERIN", "CL": "CALARASI",
        "CJ": "CLUJ", "CT": "CONSTANTA", "CV": "COVASNA", "DB": "DAMBOVITA", "DJ": "DOLJ", "GL": "GALATI",
        "GR": "GIURGIU", "GJ": "GORJ", "HR": "HARGHITA", "HD": "HUNEDOARA", "IL": "IALOMITA", "IS": "IASI",
        "IF": "ILFOV", "MM": "MARAMURES", "MH": "MEHEDINTI", "MS": "MURES", "NT": "NEAMT", "OT": "OLT",
        "PH": "PRAHOVA", "SM": "SATU MARE", "SJ": "SALAJ", "SB": "SIBIU", "SV": "SUCEAVA", "TR": "TELEORMAN",
        "TM": "TIMIS", "TL": "TULCEA", "VS": "VASLUI", "VL": "VALCEA", "VN": "VRANCEA", "B": "BUCURESTI"
    }

    import unicodedata

    def normalize_county(s):
        s = str(s)
        s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
        s = s.upper().strip()
        s = s.replace("-", " ")
        return s

    shapefile_path = r'Data\Romania1000k.shp'

    try:
        gdf = gpd.read_file(shapefile_path)
    except Exception as e:
        st.error(f"Nu s-a putut citi shapefile la {shapefile_path}. Eroare: {e}")
        st.stop()

    county_col_candidates = [c for c in gdf.columns if c.upper() in ['NUME', 'JUDET', 'COUNTY', 'COUNTY_NAME']]
    county_col = county_col_candidates[0] if county_col_candidates else gdf.columns[0]

    gdf['county_clean'] = gdf[county_col].apply(normalize_county)
    barplot_df['Judet_full'] = barplot_df['Judet'].map(COUNTY_CODE_TO_NAME)
    barplot_df['county_clean'] = barplot_df['Judet_full'].apply(normalize_county)

    map_df = gdf.set_index('county_clean').join(barplot_df.set_index('county_clean'))

    def generate_interpretation(row):
        if pd.isna(row['Difference']):
            return "Nu există date pentru acest județ."
        elif row['Difference'] > 0:
            return f"În {row[county_col].title()}, numărul de elevi a crescut cu {int(row['Difference']):,}."
        elif row['Difference'] < 0:
            return f"În {row[county_col].title()}, numărul de elevi a scăzut cu {abs(int(row['Difference'])):,}."
        else:
            return f"În {row[county_col].title()}, numărul de elevi nu s-a schimbat."

    map_df['interpretation'] = map_df.apply(generate_interpretation, axis=1)

    missing = map_df[map_df['Difference'].isnull()]
    if len(missing) == len(map_df):
        st.error(
            "Niciun județ nu a fost asociat între shapefile și date. Verifică maparea codurilor de județ!"
        )
        st.write("Județe în shapefile:", gdf['county_clean'].unique())
        st.write("Județe în date (după mapare):", barplot_df['county_clean'].unique())
        st.stop()
    elif len(missing) > 0:
        st.warning(f"Unele județe din shapefile nu au fost asociate cu date și vor apărea gri: {', '.join(missing[county_col].dropna().tolist())}")
    else:
        st.success("Toate județele au fost asociate cu succes!")

    st.write("Previzualizare date pentru hartă (primele 10 rânduri):")
    st.dataframe(map_df[[county_col, 'Judet', 'Difference', 'interpretation']].head(10))

    romania_center = [45.9432, 24.9668]
    m = folium.Map(location=romania_center, zoom_start=7, tiles="CartoDB positron")

    folium.Choropleth(
        geo_data=map_df.__geo_interface__,
        name="Abandon școlar",
        data=map_df,
        columns=[county_col, 'Difference'],
        key_on=f'feature.properties.{county_col}',
        fill_color='RdYlBu',
        fill_opacity=0.8,
        line_opacity=0.2,
        nan_fill_color='lightgray',
        legend_name=f"Schimbarea abandonului școlar ({last_delta.replace('Δ_', '').replace('→',' până la ')})",
        highlight=True,
        bins=7,
        reset=True
    ).add_to(m)

    geojson = folium.GeoJson(
        map_df,
        style_function=lambda x: {
            'fillColor': '#transparent',
            'color': '#666',
            'weight': 1,
            'fillOpacity': 0
        },
        tooltip=GeoJsonTooltip(
            fields=[county_col, 'Judet', 'Difference', 'interpretation'],
            aliases=['Județ', 'Cod scurt', 'Schimbare abandon școlar', 'Interpretare'],
            localize=True,
            sticky=False,
            labels=True,
            style=(
                "background-color: white; color: #333; font-family: Arial; font-size: 14px; "
                "border: 1px solid gray; border-radius: 3px; box-shadow:2px 2px 6px #aaa;"
            )
        )
    )
    geojson.add_to(m)

    folium.LayerControl().add_to(m)
    folium_static(m, width=820, height=600)
    #endregion

    CLEAN_PATH = "Data/cleaned_data.pkl"
    df_clean.to_pickle(CLEAN_PATH)
    st.session_state["df_clean"] = df_clean.copy()

if __name__ == "__main__":
    main()

#endregion
