import pandas as pd
import streamlit as st

ALTERNATIVES = [
    "Protocole standard",
    "Protocole intensif",
    "Approche combinee",
    "Suivi ambulatoire",
    "Programme personnalise",
]
CRITERES = [
    "Efficacite clinique (%)",
    "Risque d'effets indesirables (%)",
    "Delai d'amelioration (jours)",
    "Adherence attendue (%)",
    "Cout estime (EUR)",
]
TYPES = ["max", "min", "min", "max", "min"]

DATA = [
    [72,  8, 14, 82, 180],
    [84, 18,  7, 68, 420],
    [80, 11, 10, 78, 310],
    [63,  5, 21, 90, 120],
    [88, 12,  9, 86, 560],
]


def appliquer_style():
    """Applique l'identite visuelle commune aux pages de l'application."""
    st.markdown(
        """
        <style>
            .stApp { background: #f6f8fa; color: #17212b; }
            [data-testid="stHeader"] { background: rgba(246, 248, 250, 0.92); }
            [data-testid="stSidebar"] { background: #103b45; }
            [data-testid="stSidebar"] * { color: #f4fbfb; }
            h1 { color: #103b45; font-weight: 700; }
            h2, h3 { color: #176b72; }
            div[data-testid="stMetric"] {
                background: #ffffff;
                border-left: 4px solid #e08a3c;
                padding: 0.8rem;
            }
            .stButton > button[kind="primary"] {
                background: #176b72;
                border-color: #176b72;
            }
            .stButton > button[kind="primary"]:hover {
                background: #103b45;
                border-color: #103b45;
            }
            [data-testid="stDataFrame"] { border: 1px solid #d9e1e5; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state():
    """Initialise la matrice et les types au premier chargement.
    À appeler au début de chaque page."""
    if "matrice" not in st.session_state:
        m = pd.DataFrame(DATA, index=ALTERNATIVES, columns=CRITERES)
        m.index.name = "Alternative"
        st.session_state.matrice = m
        st.session_state.types = pd.Series(TYPES, index=CRITERES, name="Type")
def afficher_steps(res):
    for i, (titre, obj) in enumerate(res["steps"], 1):
        st.markdown(f"**Étape {i} : {titre}**")
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            df = pd.DataFrame(obj)
            st.dataframe(df.style.format("{:.4f}", subset=df.select_dtypes("number").columns),
                         use_container_width=True)
        else:
            st.write(obj)
