import streamlit as st
from methods import valider_matrice
from utils import appliquer_style, init_state

st.set_page_config(page_title="MediDecide | Cas clinique", layout="wide")
init_state()
appliquer_style()

st.title("MediDecide")
st.caption("Aide a la comparaison multicritere de protocoles de prise en charge")
st.warning(
    "Outil de demonstration pedagogique : les resultats ne constituent pas une "
    "recommandation medicale et ne remplacent jamais le jugement d'un professionnel de sante."
)
st.write("Renseignez les options de prise en charge, puis calculez les poids des criteres ou le classement dans le menu lateral.")

# --- Matrice de décision -------------------------------------------------
st.subheader("Options de prise en charge")
st.caption("Modifiez les indicateurs, renommez les protocoles ou ajoutez et supprimez des options.")

df = st.session_state.matrice.reset_index()
edite = st.data_editor(df, num_rows="dynamic", hide_index=True,
                       use_container_width=True)

# --- Types de critères ---------------------------------------------------
st.subheader("Sens clinique des criteres")
st.caption("**max** : une valeur plus elevee est preferable | **min** : une valeur plus faible est preferable")

types_df = st.session_state.types.to_frame().T
types_edite = st.data_editor(
    types_df, hide_index=True, use_container_width=True,
    column_config={
        c: st.column_config.SelectboxColumn(options=["max", "min"], required=True)
        for c in types_df.columns
    },
)

# --- Validation et enregistrement ---------------------------------------
try:
    edite = edite.dropna(subset=["Alternative"])
    if edite["Alternative"].duplicated().any():
        raise AssertionError("Les noms d'alternatives doivent être uniques.")
    if edite.isna().any().any():
        raise AssertionError("Certaines cases sont vides.")
    nouvelle = edite.set_index("Alternative")
    nouveaux_types = types_edite.iloc[0].rename("Type")

    valider_matrice(nouvelle, nouveaux_types)

    st.session_state.matrice = nouvelle
    st.session_state.types = nouveaux_types
    st.success(f"Matrice valide : {nouvelle.shape[0]} options x "
               f"{nouvelle.shape[1]} criteres. Elle est prete pour la ponderation et le classement.")
except AssertionError as e:
    st.error(f"Matrice non valide : {e}")
