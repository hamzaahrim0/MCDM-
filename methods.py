import numpy as np
import pandas as pd
from scipy.optimize import linprog

# Indice de cohérence de BWM selon a_BW (valeur de la comparaison Meilleur vs Pire)
CI_BWM = {1: 0.00, 2: 0.44, 3: 1.00, 4: 1.63, 5: 2.30,
          6: 3.00, 7: 3.73, 8: 4.47, 9: 5.23}

def minmax_normalize(matrice, types):
    """Normalisation min-max : 1 = meilleur, 0 = pire, pour chaque critère
    (inverse automatiquement les critères de type coût)."""
    mn, mx = matrice.min(), matrice.max()
    norm = (matrice - mn) / (mx - mn)
    for c in matrice.columns:
        if types[c] == "min":
            norm[c] = 1 - norm[c]
    return norm


def valider_matrice(matrice, types):
    """Vérifie que la matrice est exploitable par toutes les méthodes."""
    assert matrice.shape[0] >= 2, "Il faut au moins 2 alternatives."
    assert matrice.shape[1] >= 2, "Il faut au moins 2 critères."
    assert (matrice.dtypes.apply(lambda t: np.issubdtype(t, np.number))).all(), "Valeurs non numériques."
    assert (matrice > 0).all().all(), "Valeurs strictement positives requises (Entropie)."
    assert set(types) <= {"max", "min"}, "Types autorisés : 'max' ou 'min'."
    assert list(types.index) == list(matrice.columns), "Types et critères désalignés."



def creer_resultat(steps, result):
    """Format commun retourné par toutes les méthodes.
    steps  : liste de tuples (titre, objet DataFrame/Series/texte)
    result : poids ou classement final"""
    return {"steps": steps, "result": result}


def entropie(matrice, types=None):
    
    m = matrice.shape[0]
    steps = []

    # 1) Matrice des proportions
    P = matrice / matrice.sum()
    steps.append(("Matrice des proportions", P))

    # 2) Entropie par critère
    k = 1 / np.log(m)
    e = -k * (P * np.log(P)).sum()
    e.name = "Entropie e_j"
    steps.append((f"Entropie  e_j = -k Σ p_ij ln(p_ij)   avec k = 1/ln({m}) = {k:.4f}", e))

    # 3) Divergence
    d = (1 - e).rename("Divergence d_j")
    steps.append(("Degré de divergence  d_j = 1 - e_j", d))

    # 4) Poids
    w = (d / d.sum()).rename("Poids (Entropie)")
    steps.append(("Poids  w_j ", w))

    return creer_resultat(steps, w)



def critic(matrice, types):
    "Méthode CRITIC (poids objectifs). Utilise minmax_normalize défini dans le socle."
    steps = []

    # 1) Normalisation min-max (coûts inversés)
    N = minmax_normalize(matrice, types)
    steps.append(("Matrice normalisée (min-max, coûts inversés)", N))

    # 2) Écart-type par critère (échantillon, ddof=1)
    sigma = N.std(ddof=1).rename("Écart-type σ_j")
    steps.append(("Écart-type de chaque critère  σ_j", sigma))

    # 3) Matrice de corrélation
    R = N.corr()
    steps.append(("Matrice de corrélation  r_jk", R))

    # 4) Conflit
    conflit = (1 - R).sum().rename("Conflit Σ(1 - r_jk)")
    steps.append(("Conflit  Σ_k (1 - r_jk)", conflit))

    # 5) Quantité d'information
    C = (sigma * conflit).rename("Information C_j")
    steps.append(("Quantité d'information  C_j = σ_j × Σ_k (1 - r_jk)", C))

    # 6) Poids
    w = (C / C.sum()).rename("Poids (CRITIC)")
    steps.append(("Poids  w_j = C_j / Σ C_j", w))

    return creer_resultat(steps, w)


def construire_matrice_ahp(criteres, comparaisons):
    """Construit la matrice réciproque à partir du triangle supérieur.
    comparaisons : dict {(critère_i, critère_j): valeur de Saaty} avec i avant j."""
    n = len(criteres)
    A = pd.DataFrame(np.ones((n, n)), index=criteres, columns=criteres)
    for (ci, cj), v in comparaisons.items():
        A.loc[ci, cj] = v
        A.loc[cj, ci] = 1 / v
    return A


# Indice aléatoire de Saaty (RI) selon la taille n de la matrice
RI_TABLE = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
            6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}


def ahp(A):
    """Méthode AHP (poids subjectifs).
    A : DataFrame carré, matrice de comparaisons par paires (réciproque)."""
    n = A.shape[0]
    assert A.shape[0] == A.shape[1], "La matrice doit être carrée."
    assert np.allclose(A.values, 1 / A.values.T), "La matrice doit être réciproque."
    steps = []

    # 1) Matrice de comparaisons
    steps.append(("Matrice de comparaisons par paires  A", A))

    # 2) Normalisation par colonne + moyenne par ligne (approximation)
    N = A / A.sum()
    steps.append(("Matrice normalisée (chaque colonne divisée par sa somme)", N))
    w_approx = N.mean(axis=1).rename("Poids approchés")
    steps.append(("Poids approchés (moyenne de chaque ligne)", w_approx))

    # 3) Poids exacts : vecteur propre principal
    valeurs, vecteurs = np.linalg.eig(A.values)
    idx = np.argmax(valeurs.real)
    lambda_max = valeurs[idx].real
    v = np.abs(vecteurs[:, idx].real)
    w = pd.Series(v / v.sum(), index=A.index, name="Poids (AHP)")
    steps.append(("Poids exacts (vecteur propre principal, normalisé)", w))

    # 4) Indice de cohérence
    CI = (lambda_max - n) / (n - 1)
    steps.append((f"λ_max = {lambda_max:.4f}   →   CI = (λ_max - n)/(n - 1) = {CI:.4f}",
                  pd.Series({"λ_max": lambda_max, "CI": CI})))

    # 5) Ratio de cohérence
    RI = RI_TABLE[n]
    CR = CI / RI if RI > 0 else 0.0
    verdict = " Jugements cohérents (CR < 0.1)" if CR < 0.1 else " Jugements incohérents (CR ≥ 0.1), à revoir"
    steps.append((f"Ratio de cohérence  CR = CI / RI = {CI:.4f} / {RI} = {CR:.4f}   →   {verdict}",
                  pd.Series({"RI": RI, "CR": CR})))

    return creer_resultat(steps, w)



def bwm(criteres, best, worst, best_to_others, others_to_worst):
    
    n = len(criteres)
    iB, iW = criteres.index(best), criteres.index(worst)
    assert best != worst, "Le meilleur et le pire critère doivent être différents."
    assert best_to_others[best] == 1 and others_to_worst[worst] == 1, \
        "a_BB et a_WW doivent valoir 1."
    assert best_to_others[worst] == others_to_worst[best], \
        "La comparaison Meilleur/Pire doit être identique dans les deux vecteurs."
    steps = []

    # 1) Vecteurs de comparaison
    tableau = pd.DataFrame(
        {"Best-to-Others (a_Bj)": pd.Series(best_to_others),
         "Others-to-Worst (a_jW)": pd.Series(others_to_worst)}
    ).loc[criteres]
    steps.append((f"Jugements du décideur  (Meilleur = {best}, Pire = {worst})", tableau))
    # 2) Programme linéaire : variables x = [w_1..w_n, ξ], on minimise ξ
    c = np.zeros(n + 1); c[-1] = 1
    A_ub, b_ub = [], []
    for j, cj in enumerate(criteres):
        # |w_B - a_Bj w_j| <= ξ
        ligne = np.zeros(n + 1); ligne[iB] += 1; ligne[j] -= best_to_others[cj]
        A_ub.append(np.append(ligne[:-1], -1)); b_ub.append(0)
        A_ub.append(np.append(-ligne[:-1], -1)); b_ub.append(0)
        # |w_j - a_jW w_W| <= ξ
        ligne = np.zeros(n + 1); ligne[j] += 1; ligne[iW] -= others_to_worst[cj]
        A_ub.append(np.append(ligne[:-1], -1)); b_ub.append(0)
        A_ub.append(np.append(-ligne[:-1], -1)); b_ub.append(0)
    A_eq = [np.append(np.ones(n), 0)]; b_eq = [1]

    sol = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=[(0, None)] * (n + 1), method="highs")
    assert sol.success, "Le programme linéaire n'a pas de solution."

    w = pd.Series(sol.x[:n], index=criteres, name="Poids (BWM)")
    xi = sol.x[-1]
    steps.append(("Résolution du programme linéaire : poids optimaux", w))
    # 3) Cohérence
    a_BW = best_to_others[worst]
    CI = CI_BWM[int(a_BW)]
    CR = xi / CI if CI > 0 else 0.0
    verdict = "Jugements cohérents" if CR < 0.1 else "Cohérence à surveiller (CR élevé)"
    steps.append((f"ξ* = {xi:.4f},  a_BW = {int(a_BW)},  CI = {CI}  →  CR = ξ*/CI = {CR:.4f}   {verdict}",
                  pd.Series({"ξ*": xi, "CI": CI, "CR": CR})))

    return creer_resultat(steps, w)


def normalisation_lineaire(matrice, types):
    """Normalisation par rapport au meilleur : valeurs dans ]0, 1], 1 = meilleur."""
    N = matrice.astype(float).copy()
    for c in matrice.columns:
        if types[c] == "max":
            N[c] = matrice[c] / matrice[c].max()
        else:
            N[c] = matrice[c].min() / matrice[c]
    return N

def waspas(matrice, types, poids, lam=0.5):
    """Méthode WASPAS (classement).
    poids : Series indexée par les critères, de somme 1
    lam   : paramètre λ entre 0 (WPM pur) et 1 (WSM pur)"""
    assert 0 <= lam <= 1, "λ doit être compris entre 0 et 1."
    assert list(poids.index) == list(matrice.columns), "Poids et critères désalignés."
    assert np.isclose(poids.sum(), 1), "La somme des poids doit valoir 1."
    steps = []

    # 1) Poids utilisés
    steps.append(("Poids des critères utilisés", poids.rename("Poids")))

    # 2) Normalisation linéaire
    N = normalisation_lineaire(matrice, types)
    steps.append(("Matrice normalisée (bénéfice : x/max, coût : min/x)", N))

    # 3) WSM
    wsm = (N * poids).sum(axis=1).rename("WSM")
    steps.append(("Somme pondérée  Q1_i = Σ_j w_j r_ij", wsm))

    # 4) WPM
    wpm = (N ** poids).prod(axis=1).rename("WPM")
    steps.append(("Produit pondéré  Q2_i = Π_j r_ij^w_j", wpm))

    # 5) Score final
    Q = (lam * wsm + (1 - lam) * wpm).rename("Q")
    steps.append((f"Score final  Q_i = λ·Q1_i + (1-λ)·Q2_i   avec λ = {lam}", Q))

    # 6) Classement
    classement = pd.DataFrame({"WSM": wsm, "WPM": wpm, "Q": Q})
    classement["Rang"] = classement["Q"].rank(ascending=False, method="min").astype(int)
    classement = classement.sort_values("Rang")

    return creer_resultat(steps, classement)


def topsis(matrice, types, poids):
    """Méthode TOPSIS (classement).
    poids : Series indexée par les critères, de somme 1"""
    assert list(poids.index) == list(matrice.columns), "Poids et critères désalignés."
    assert np.isclose(poids.sum(), 1), "La somme des poids doit valoir 1."
    steps = []

    # 1) Poids utilisés
    steps.append(("Poids des critères utilisés", poids.rename("Poids")))

    # 2) Normalisation vectorielle
    N = matrice / np.sqrt((matrice ** 2).sum())
    steps.append(("Matrice normalisée  r_ij = x_ij / √(Σ x_ij²)", N))

    # 3) Matrice pondérée
    V = N * poids
    steps.append(("Matrice pondérée  v_ij = w_j · r_ij", V))

    # 4) Solutions idéale et anti-idéale
    ideal = pd.Series(
        {c: V[c].max() if types[c] == "max" else V[c].min() for c in V.columns},
        name="Idéale A+")
    anti = pd.Series(
        {c: V[c].min() if types[c] == "max" else V[c].max() for c in V.columns},
        name="Anti-idéale A-")
    steps.append(("Solutions idéale A+ et anti-idéale A-", pd.concat([ideal, anti], axis=1).T))

    # 5) Distances
    d_plus = np.sqrt(((V - ideal) ** 2).sum(axis=1)).rename("D+")
    d_moins = np.sqrt(((V - anti) ** 2).sum(axis=1)).rename("D-")
    steps.append(("Distances à l'idéal (D+) et à l'anti-idéal (D-)",
                  pd.concat([d_plus, d_moins], axis=1)))

    # 6) Coefficient de proximité
    C = (d_moins / (d_plus + d_moins)).rename("C")
    steps.append(("Coefficient de proximité  C_i = D- / (D+ + D-)", C))

    # 7) Classement
    classement = pd.DataFrame({"D+": d_plus, "D-": d_moins, "C": C})
    classement["Rang"] = classement["C"].rank(ascending=False, method="min").astype(int)
    classement = classement.sort_values("Rang")

    return creer_resultat(steps, classement)