import os
import glob
import pandas as pd
import streamlit as st
import geopandas as gpd
from utils import ordinal_encode_nivel, binary_encode_mediu, normalize_county

# region: Load Raw Data ######################################################
@st.cache_data
def load_raw_data(data_dir: str = "Data") -> pd.DataFrame:
    """
    Load all student CSVs from a directory, tag each with its period, and concatenate.
    """
    pattern = os.path.join(data_dir, "*_elevi.csv")
    files = glob.glob(pattern)
    dfs = []
    for fp in files:
        period = os.path.splitext(os.path.basename(fp))[0]
        df = pd.read_csv(fp)
        df["period"] = period
        dfs.append(df)
    if dfs:
        combined = pd.concat(dfs, ignore_index=True, sort=False)
    else:
        combined = pd.DataFrame()
    return combined

# region: Full Pipeline ######################################################
@st.cache_data
def load_and_prepare(data_dir: str = "Data", geo_shp: str = None) -> pd.DataFrame:
    """
    One-stop function: load raw CSVs, clean/harmonize, encode/impute, and engineer features.
    """
    raw = load_raw_data(data_dir)
    clean = clean_and_harmonize(raw)
    encoded = encode_and_impute(clean)
    final = engineer_features(encoded, geo_shp)
    return final

# region: Initial Cleaning & Harmonization ##################################
@st.cache_data
def clean_and_harmonize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names, drop sparsely-populated columns, and coerce IDs.
    """
    # 1. Snake-case column names
    df.columns = (
        df.columns
          .str.strip()
          .str.lower()
          .str.replace(r"[ \-]+", "_", regex=True)
          .str.normalize('NFKD')
          .str.encode('ascii', errors='ignore')
          .str.decode('utf-8')
    )
    # 2. Drop columns with >95% missing values
    thresh = len(df) * 0.05
    df = df.dropna(axis=1, thresh=thresh)
    # 3. Coerce ID columns to integer
    for col in ['_id', 'cod_unitate_pj', 'cod_unitate_plan']:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
    return df

# region: Get Clean Data (for EDA pages) ####################################
@st.cache_data(show_spinner=False)
def get_clean_data() -> pd.DataFrame:
    CLEAN_PATH = "Data/cleaned_data.pkl"
    if os.path.exists(CLEAN_PATH):
        df = pd.read_pickle(CLEAN_PATH)
    elif "df_clean" in st.session_state:
        df = st.session_state["df_clean"].copy()
    else:
        st.warning("Cleaned data not found. Please run the Exploratory Analysis page first.")
        df = load_raw_data()

    if "Elevi exist anterior-asoc" in df.columns:
        df = df.rename(columns={"Elevi exist anterior-asoc": "elevi_existenti"})

    return df
# region: Shapefile Loader ###################################################
@st.cache_data
def load_shapefile(path: str, tolerance: float = 0.01) -> gpd.GeoDataFrame:
    """
    Load and simplify a GeoDataFrame from a shapefile for faster mapping.
    """
    gdf = gpd.read_file(path)
    gdf['geometry'] = gdf['geometry'].simplify(tolerance, preserve_topology=True)
    return gdf

# region: Encoding & Imputation ##############################################
@st.cache_data
def encode_and_impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply custom encodings for ordinal and binary fields, fill missing values.
    """
    # Ordinal encode performance level
    if 'nivel_de_performanta' in df.columns:
        df['nivel_enc'] = ordinal_encode_nivel(df['nivel_de_performanta'])
    # Binary encode environment
    if 'mediu' in df.columns:
        df['mediu_enc'] = binary_encode_mediu(df['mediu'])
    # Categorical imputation: fill NaNs with 'Unknown'
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col] = df[col].fillna('Unknown')
    # Numeric imputation: fill NaNs with median
    num_cols = df.select_dtypes(include=['float','Int64']).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    return df

# region: Feature Engineering ################################################
@st.cache_data
def engineer_features(df: pd.DataFrame, geo_shp: str) -> pd.DataFrame:
    """
    Derive dropout rates, year-over-year changes, and county-level aggregates.
    """
    # 1. Compute dropout_rate if columns exist
    if 'suplimentar' in df.columns and 'numarul_elevilor_existenti' in df.columns:
        df['dropout_rate'] = df['suplimentar'] / df['numarul_elevilor_existenti']
    # 2. Year-over-year percent change per school
    if 'dropout_rate' in df.columns and 'cod_unitate_pj' in df.columns and 'period' in df.columns:
        df = df.sort_values(['cod_unitate_pj','period'])
        df['yoy_dropout_pct'] = df.groupby('cod_unitate_pj')['dropout_rate'].pct_change() * 100
    # 3. Normalize and merge county aggregates
    if geo_shp:
        gdf = gpd.read_file(geo_shp)
        gdf['county_norm'] = normalize_county(gdf['NAME_1'])
        df['county_norm'] = normalize_county(df['judet'])
        agg = (
            df.groupby(['period','county_norm'])['dropout_rate']
              .mean()
              .reset_index()
              .rename(columns={'dropout_rate':'avg_dropout_rate'})
        )
        df = df.merge(agg, on=['period','county_norm'], how='left')
    return df

# region: Model Data Preparation #############################################
@st.cache_data
def get_model_data(csv_paths: list, geo_shp: str):
    """
    Return feature matrix X and target y for modeling dropout_rate.
    """
    df = load_and_prepare()  # uses default Data directory and full pipeline
    # Drop non-feature columns
    drop_cols = [
        '_id','cod_unitate_pj','cod_unitate_plan',
        'denumire_unitate_pj','denumire_unitate_plan',
        'judet','county_norm','period'
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df])
    df = df[df['dropout_rate'].notnull()]
    X = df.drop(columns=['dropout_rate'])
    y = df['dropout_rate']
    return X, y

@st.cache_data(show_spinner=False)
def get_demographics_data(path: str = "Data/demographics.csv") -> pd.DataFrame:
    """
    Load county-level demographics, map county names to standardized codes.
    """
    df = pd.read_csv(path)
    # Clean column names
    df.columns = df.columns.str.strip()
    # Rename county column
    county_col = 'Macroregiuni  regiuni de dezvoltare si judete'
    if county_col in df.columns:
        df = df.rename(columns={county_col: 'Judet'})
    # Strip whitespace
    df['Judet'] = df['Judet'].astype(str).str.strip()
    # Normalize county names
    df['Judet_norm'] = df['Judet'].apply(lambda s: normalize_county(s))
    # Map full names to codes
    name_to_code = {
        'ALBA': 'Ab', 'ARAD': 'Ar', 'ARGES': 'Ag', 'BACAU': 'Bc', 'BIHOR':'Bh',
        'BISTRITA NASAUD':'Bn', 'BRAILA':'Br','BOTOSANI':'Bt', 'BRASOV':'Bv',
        'BUZAU':'Bz','CARAS SEVERIN':'Cs', 'CALARASI':'Cl', 'CLUJ':'Cj',
        'CONSTANTA':'Ct', 'COVASNA':'Cv', 'DAMBOVITA':'Db','DOLJ':'Dj',
        'GALATI':'Gl','GIURGIU':'Gr', 'GORJ':'Gj', 'HARGHITA':'Hr',
        'HUNEDOARA':'Hd','IALOMITA':'Il', 'IASI':'Is','ILFOV':'If',
        'MARAMURES':'Mm', 'MEHEDINTI':'Mh','MURES':'Ms','NEAMT':'Nt',
        'OLT':'Ot','PRAHOVA':'Ph','SATU MARE':'Sm','SALAJ':'Sj','SIBIU':'Sb',
        'SUCEAVA':'Sv','TELEORMAN':'Tr','TIMIS':'Tm','TULCEA':'Tl','VALCEA':'Vl',
        'VASLUI':'Vs','VRANCEA':'Vn','BUCURESTI':'B','UNKNOWN':'Nan'
    }
    df['Judet_code'] = df['Judet_norm'].map(name_to_code).fillna('Nan')

    age_groups = {
        '0 ani': '0-3',
        '1 ani': '0-3',
        '2 ani': '0-3',
        '3 ani': '3-6',
        '4 ani': '3-6',
        '5 ani': '3-6',
        '6 ani': '6-11',
        '7 ani': '6-11',
        '8 ani': '6-11',
        '9 ani': '6-11',
        '10 ani': '11-14',
        '11 ani': '11-14',
        '12 ani': '11-14',
        '13 ani': '11-14',
        '14 ani': '14-18',
        '15 ani': '14-18',
        '16 ani': '14-18',
        '17 ani': '14-18',
        '18 ani': '>18',
        '19 ani': '>18',

        '0- 4 ani': 'other',
        '5- 9 ani': 'other',
        '10-14 ani': 'other',
        '15-19 ani': 'other',
    }
    df['age_group'] = df['Varste si grupe de varsta'].map(age_groups).fillna('other')

    df = df[df['age_group'] != 'other']
    df = df[df['Judet_code'] != 'Nan']

    # Extract year from period
    # Extract year, keep as float (allows NaN), then drop rows where year is missing, then cast to int
    df['year'] = df['Perioade'].str.strip().str.extract(r'Anul (\d{4})')
    df = df[df['year'].notnull()]  # Drop rows without a valid year
    df['year'] = df['year'].astype(int)

    df['Valoare'] = pd.to_numeric(df['Valoare'], errors='coerce')
    df.dropna(subset=['Valoare'], inplace=True)

    # Filter for target years (as available)
    df_filtered = df[df['year'].isin([2021, 2022, 2023])].copy()

    # Pivot to get one row per (Judet_code, age_group, [sex, mediu, ...]) per year
    id_vars = ['Judet', 'Judet_norm', 'Judet_code', 'age_group', 'Sexe', 'Medii de rezidenta']
    df_pivot = df_filtered.pivot_table(
        index=id_vars, columns='year', values='Valoare'
    ).reset_index()
    df_pivot.columns.name = None

    # Compute year-on-year % change and absolute change
    for (y1, y2) in [(2021, 2022), (2022, 2023)]:
        c1, c2 = str(y1), str(y2)
        if y1 in df_pivot.columns and y2 in df_pivot.columns:
            df_pivot[f'change_{y1}_{y2}'] = df_pivot[y2] - df_pivot[y1]
            df_pivot[f'change_{y1}_{y2}'] = (
                (df_pivot[y2] - df_pivot[y1]) / df_pivot[y1].replace(0, 1e-6) * 100
            )
        else:
            df_pivot[f'change_{y1}_{y2}'] = pd.NA
            df_pivot[f'change_{y1}_{y2}'] = pd.NA

    # Optional: average percent change across both periods
    pct_cols = [f'change_{y1}_{y2}' for (y1, y2) in [(2021, 2022), (2022, 2023)]]
    df_pivot['change_pct'] = df_pivot[pct_cols].mean(axis=1)

    return df_pivot


