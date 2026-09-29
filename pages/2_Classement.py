import streamlit as st
import pandas as pd
import numpy as np
from methods import waspas, topsis
from utils import appliquer_style, init_state, afficher_steps

st.set_page_config(page_title="MediDecide | Classement", layout="wide")
init_state()
appliquer_style()

matrice = st.session_state.matrice
types = st.session_state.types
criteres = list(matrice.columns)

st.title("Classement des options")
st.caption("Comparez les protocoles selon les criteres et les poids retenus.")

methode = st.sidebar.radio("Méthode", ["WASPAS", "TOPSIS"])
st.subheader(f"Méthode : {methode}")

# --- 1. Choix des poids --------------------------------------------------
st.markdown("#### Poids des criteres")

# Les poids mémorisés ne sont utilisables que s'ils correspondent aux critères actuels
poids_memo = st.session_state.get("poids")
memo_ok = poids_memo is not None and list(poids_memo.index) == criteres

options = ["Saisie manuelle"]
if memo_ok:
    options.insert(0, f"Derniers poids calcules ({st.session_state.poids_source})")
origine = st.radio("Origine des poids", options, horizontal=True)

if origine == "Saisie manuelle":
    st.caption("Saisissez des valeurs positives. Elles seront normalisees pour que leur somme vaille 1.")
    brut = st.data_editor(
        pd.DataFrame({"Poids": [1.0] * len(criteres)}, index=criteres),
        use_container_width=True,
        key="poids_manuels",
    )["Poids"]
    if (brut <= 0).any() or brut.isna().any():
        st.error("Tous les poids doivent être strictement positifs.")
        st.stop()
    poids = (brut / brut.sum()).rename("Poids")
else:
    poids = poids_memo.rename("Poids")
    if not memo_ok:
        st.warning("Aucun poids calcule : utilisez la page Ponderation ou la saisie manuelle.")

st.dataframe(poids.to_frame().T.style.format("{:.4f}"), use_container_width=True)

# --- 2. Paramètres -------------------------------------------------------
lam = 0.5
if methode == "WASPAS":
    st.markdown("#### Parametre lambda")
    lam = st.slider("λ (0 = produit pondéré pur, 1 = somme pondérée pure)",
                    0.0, 1.0, 0.5, 0.05)

# --- 3. Run --------------------------------------------------------------
if st.button("Classer les options", type="primary"):
    try:
        if methode == "WASPAS":
            res = waspas(matrice, types, poids, lam)
        else:
            res = topsis(matrice, types, poids)
        st.session_state.res_class = {"methode": methode, "res": res}
    except AssertionError as e:
        st.error(f"Erreur : {e}")

# --- Affichage du dernier résultat --------------------------------------
stocke = st.session_state.get("res_class")
if stocke and stocke["methode"] == methode:
    st.divider()
    afficher_steps(stocke["res"])

    st.markdown("### Resultat : classement")
    classement = stocke["res"]["result"]
    score = "Q" if methode == "WASPAS" else "C"

    meilleur = classement.index[0]
    st.success(f"Option la mieux classee : **{meilleur}** ({score} = {classement[score].iloc[0]:.4f})")

    g, t = st.columns([2, 1])
    g.bar_chart(classement[score])
    t.dataframe(classement.style.format({c: "{:.4f}" for c in classement.columns if c != "Rang"}),
                use_container_width=True)
