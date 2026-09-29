import streamlit as st
import pandas as pd
from methods import entropie, critic, ahp, bwm, construire_matrice_ahp
from utils import appliquer_style, init_state, afficher_steps

st.set_page_config(page_title="MediDecide | Ponderation", layout="wide")
init_state()
appliquer_style()

matrice = st.session_state.matrice
types = st.session_state.types
criteres = list(matrice.columns)

st.title("Ponderation des criteres")
st.caption("Definissez l'importance relative des criteres cliniques avant de comparer les options.")

methode = st.sidebar.radio(
    "Méthode",
    ["Entropie", "CRITIC", "AHP", "BWM"],
    captions=["Basee sur les donnees", "Basee sur les donnees", "Jugement expert", "Jugement expert"],
)
st.subheader(f"Méthode : {methode}")

# --- Saisie des paramètres (méthodes subjectives) ------------------------
A = None
if methode in ("Entropie", "CRITIC"):
    st.info("Methode basee sur les donnees : les poids sont calcules a partir de la matrice clinique.")
    st.dataframe(matrice, use_container_width=True)

elif methode == "AHP":
    if len(criteres) > 10:
        st.error("AHP est limité à 10 critères (table de l'indice aléatoire).")
        st.stop()
    st.caption("Pour chaque paire, indiquez l'importance du 1er critère par rapport au 2e "
               "(échelle de Saaty). **3** = le 1er est modérément plus important ; "
               "**1/3** = c'est le 2e qui l'est.")
    options = [1 / k for k in range(9, 1, -1)] + [float(k) for k in range(1, 10)]
    fmt = lambda v: str(int(v)) if v >= 1 else f"1/{round(1 / v)}"

    comparaisons = {}
    colonnes = st.columns(2)
    k = 0
    for i in range(len(criteres)):
        for j in range(i + 1, len(criteres)):
            with colonnes[k % 2]:
                comparaisons[(criteres[i], criteres[j])] = st.select_slider(
                    f"{criteres[i]}  vs  {criteres[j]}",
                    options=options, value=1.0, format_func=fmt,
                    key=f"ahp_{i}_{j}",
                )
            k += 1
    A = construire_matrice_ahp(criteres, comparaisons)

elif methode == "BWM":
    st.caption("Choisissez le meilleur et le pire critère, puis comparez-les aux autres "
               "(échelle de 1 à 9).")
    c1, c2 = st.columns(2)
    best = c1.selectbox("Meilleur critère", criteres, index=0)
    worst = c2.selectbox("Pire critère", criteres, index=len(criteres) - 1)

    if best == worst:
        st.error("Le meilleur et le pire critère doivent être différents.")
        st.stop()

    a_BW = st.slider(f"{best} (meilleur) vs {worst} (pire)", 2, 9, 5, key=f"bw_{best}_{worst}")
    best_to_others = {best: 1, worst: a_BW}
    others_to_worst = {worst: 1, best: a_BW}

    autres = [c for c in criteres if c not in (best, worst)]
    col_b, col_w = st.columns(2)
    for c in autres:
        best_to_others[c] = col_b.slider(f"« {best} » est … plus important que « {c} »",
                                         1, 9, 2, key=f"bo_{best}_{worst}_{c}")
        others_to_worst[c] = col_w.slider(f"« {c} » est … plus important que « {worst} »",
                                          1, 9, 2, key=f"ow_{best}_{worst}_{c}")

# --- Run -----------------------------------------------------------------
if st.button("Calculer les poids", type="primary"):
    try:
        if methode == "Entropie":
            res = entropie(matrice, types)
        elif methode == "CRITIC":
            res = critic(matrice, types)
        elif methode == "AHP":
            res = ahp(A)
        else:
            res = bwm(criteres, best, worst, best_to_others, others_to_worst)
        st.session_state.res_pond = {"methode": methode, "res": res}
        st.session_state.poids = res["result"]
        st.session_state.poids_source = methode
    except AssertionError as e:
        st.error(f"Erreur : {e}")

# --- Affichage du dernier résultat --------------------------------------
stocke = st.session_state.get("res_pond")
if stocke and stocke["methode"] == methode:
    st.divider()
    afficher_steps(stocke["res"])

    st.markdown("### Resultat : poids des criteres")
    w = stocke["res"]["result"]
    g, t = st.columns([2, 1])
    g.bar_chart(w)
    t.dataframe(w.to_frame().style.format("{:.4f}"), use_container_width=True)
    st.success(f"Poids {methode} enregistres : ils seront proposes dans la page Classement.")
