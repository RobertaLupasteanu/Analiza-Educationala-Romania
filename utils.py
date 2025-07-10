import unicodedata
import pandas as pd 
import numpy as np 

COUNTY_CODE_TO_NAME = {
    "AB": "ALBA", "AR": "ARAD", "AG": "ARGES", "BC": "BACAU", "BH": "BIHOR", "BN": "BISTRITA NASAUD",
    "BR": "BRAILA", "BT": "BOTOSANI", "BV": "BRASOV", "BZ": "BUZAU", "CS": "CARAS SEVERIN", "CL": "CALARASI",
    "CJ": "CLUJ", "CT": "CONSTANTA", "CV": "COVASNA", "DB": "DAMBOVITA", "DJ": "DOLJ", "GL": "GALATI",
    "GR": "GIURGIU", "GJ": "GORJ", "HR": "HARGHITA", "HD": "HUNEDOARA", "IL": "IALOMITA", "IS": "IASI",
    "IF": "ILFOV", "MM": "MARAMURES", "MH": "MEHEDINTI", "MS": "MURES", "NT": "NEAMT", "OT": "OLT",
    "PH": "PRAHOVA", "SM": "SATU MARE", "SJ": "SALAJ", "SB": "SIBIU", "SV": "SUCEAVA", "TR": "TELEORMAN",
    "TM": "TIMIS", "TL": "TULCEA", "VS": "VASLUI", "VL": "VALCEA", "VN": "VRANCEA", "B": "BUCURESTI"
}

def normalize_county(s):
    """Normalize county name for robust matching."""
    s = str(s)
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    s = s.upper().strip().replace("-", " ")
    return s


NIVEL_ORDER = {
    "ANTEPREȘCOLAR": 0,
    "PREȘCOLAR": 1,
    "PRIMAR": 2,
    "GIMNAZIAL": 3,
    "LICEAL": 4,
    "PROFESIONAL": 5,
    "POSTLICEAL": 6,
    "CLUBUL COPIILOR": 7,
    "CLUB SPORTIV ȘCOLAR": 8,
    "PALATUL COPIILOR": 9,
    "": -1 # Handle empty strings/unknowns for Nivel
}

def ordinal_encode_nivel(s):
    """
    Encodes the 'Nivel' (Level) categorical variable into an ordinal numerical value.
    Handles various string formats, including diacritics and empty strings.
    Assigns -1 for unmapped or missing values.
    """
    if pd.isna(s):
        return NIVEL_ORDER.get("", -1) # Map NaN to the defined value for empty/unknown
    
    s = str(s).strip()
    # Normalize the string similar to how county names are normalized, but keep case if needed for dict lookup
    s_normalized = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn').upper().strip()

    # Get the ordinal value from the defined order, default to -1 if not found
    return NIVEL_ORDER.get(s_normalized, NIVEL_ORDER.get("", -1)) # Fallback to empty string value if not found


MEDIUL_ENCODING = {
    "RURAL": 0,
    "URBAN": 1,
    "": 2, 
    "NAN": 2 
}

def binary_encode_mediu(s):
    """
    Encodes the 'Mediu' (Environment) categorical variable into a binary-like numerical value.
    Maps 'Rural' to 0, 'Urban' to 1, and other/missing values to 2.
    """
    if pd.isna(s):
        return MEDIUL_ENCODING.get("", 2) # Map NaN to the defined value for empty/unknown
    
    s = str(s).strip()
    s_upper = s.upper()

    return MEDIUL_ENCODING.get(s_upper, MEDIUL_ENCODING.get("", 2))

