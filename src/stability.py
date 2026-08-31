"""A magyarázatok stabilitásának mérése (4. hét).

Futtatás:  py -m src.stability

Ez a kutatás magja. Három dolgot mérünk:

1. ZAJSTABILITÁS
   Kis, üzletileg reális zajt teszünk a numerikus jellemzőkre (a valóságban
   is pontatlanok az adatok), és megnézzük, mennyire változik a magyarázat.
   Ha egy 2%-os adatpontatlanság átrendezi a top-5 listát, akkor a
   dashboardon megjelenő "fő ok" nem megbízható.

2. MÓDSZEREK EGYETÉRTÉSE
   Ugyanaz a modell, ugyanaz az ügyfél, SHAP vs LIME. Ha eltérnek, a
   felhasználó a magyarázó megválasztásától függően más következtetést von le.

3. A LIME ÖNMAGÁVAL VALÓ EGYETÉRTÉSE
   A LIME véletlen perturbációkkal dolgozik. Ugyanarra az esetre, ugyanazzal
   a modellel, más véletlen maggal más magyarázatot adhat. Ez a módszer
   belső zaja - el kell különíteni a modell tulajdonságaitól.

Két metrikát használunk:
  - Spearman-rangkorreláció: a TELJES sorrend mennyire marad meg (-1..1)
  - top-k átfedés: a k legfontosabb jellemzőből hány közös (0..1)

A rangkorreláció finomabb, az átfedés viszont azt méri, amit a felhasználó
ténylegesen lát a képernyőn. Ezért mindkettőt jelentjük.

Kimenet:
  reports/stability_demo.csv               nyers mérési eredmények
  reports/figures/09_stability_demo.png    összefoglaló ábra
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src import config
from src.explain import (
    MODEL_LABELS,
    aggregate_to_original,
    build_lime_explainer,
    lime_values,
    load_data,
    load_pipeline,
    shap_values,
)

# Csak a valódi folytonos jellemzőket zajosítjuk. A SeniorCitizen bináris,
# azon a "kis zaj" értelmezhetetlen lenne.
PERTURB_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

# Zajszintek a jellemző szórásának arányában. 1% = mérési pontatlanság,
# 5% = elavult adat, 10% = komoly adatminőségi probléma.
NOISE_LEVELS = [0.01, 0.05, 0.10]

# Hány esetet vizsgálunk a demóban. Az 5. héten ez 100-200 lesz.
N_CASES_DEMO = 10
N_REPEATS = 3          # zajszintenként hány ismétlés


# ---------------------------------------------------------------------------
# Zajosítás
# ---------------------------------------------------------------------------

def perturb(
    X: pd.DataFrame,
    sigma_pct: float,
    stds: pd.Series,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Gauss-zaj hozzáadása a folytonos jellemzőkhöz.

    A zaj mértéke a jellemző szórásának adott százaléka - így a tenure
    (0-72 hónap) és a TotalCharges (0-8600 dollár) arányosan azonos
    mértékű zavart kap. Abszolút zaj esetén az egyiket szétvernénk, a
    másikat meg sem karcolnánk.
    """
    X_noisy = X.copy()
    for col in PERTURB_FEATURES:
        noise = rng.normal(0, sigma_pct * stds[col], size=len(X))
        X_noisy[col] = (X_noisy[col] + noise).clip(lower=0)
    return X_noisy


# ---------------------------------------------------------------------------
# Metrikák
# ---------------------------------------------------------------------------

def rank_correlation(a: pd.Series, b: pd.Series) -> float:
    """Spearman-korreláció két magyarázat között.

    A magyarázatokat ABSZOLÚT ÉRTÉK szerint rangsoroljuk, mert a
    felhasználót az érdekli, mely tényezők a legfontosabbak - nem az,
    hogy melyik irányba tolnak.
    """
    common = a.index.intersection(b.index)
    if len(common) < 3:
        return np.nan
    rho, _ = spearmanr(a[common].abs(), b[common].abs())
    return float(rho)


def top_k_overlap(a: pd.Series, b: pd.Series, k: int = 5) -> float:
    """A k legfontosabb jellemzőből hány közös, arányban."""
    set_a = set(a.abs().sort_values(ascending=False).head(k).index)
    set_b = set(b.abs().sort_values(ascending=False).head(k).index)
    return len(set_a & set_b) / k


def top1_agreement(a: pd.Series, b: pd.Series) -> int:
    """Ugyanaz-e a LEGFONTOSABB tényező. Ez az, amit egy BI dashboard
    fejlécében kiemelnénk - ha ez ingadozik, az közvetlen üzleti kockázat."""
    return int(a.abs().idxmax() == b.abs().idxmax())


# ---------------------------------------------------------------------------
# 1. mérés: zajstabilitás
# ---------------------------------------------------------------------------

def measure_noise_stability(
    model_name: str,
    X_train: pd.DataFrame,
    cases: pd.DataFrame,
    method: str = "shap",
) -> pd.DataFrame:
    """Minden esetre: eredeti magyarázat vs zajosított magyarázat."""
    stds = X_train[PERTURB_FEATURES].std()
    original_cols = list(X_train.columns)

    if method == "shap":
        base_raw = shap_values(model_name, X_train, cases)
    else:
        explainer = build_lime_explainer(model_name, X_train)
        base_raw = lime_values(model_name, X_train, cases, explainer=explainer)

    rows = []
    for sigma in NOISE_LEVELS:
        for rep in range(N_REPEATS):
            # Minden (zajszint, ismétlés) párhoz külön, de rögzített mag
            rng = np.random.default_rng(config.RANDOM_STATE + rep)
            noisy = perturb(cases, sigma, stds, rng)

            if method == "shap":
                pert_raw = shap_values(model_name, X_train, noisy)
            else:
                pert_raw = lime_values(model_name, X_train, noisy,
                                       explainer=explainer)

            for i in range(len(cases)):
                a = aggregate_to_original(base_raw.iloc[i], original_cols)
                b = aggregate_to_original(pert_raw.iloc[i], original_cols)
                rows.append({
                    "modell": model_name,
                    "modszer": method,
                    "zajszint": sigma,
                    "ismetles": rep,
                    "eset": int(cases.index[i]),
                    "spearman": rank_correlation(a, b),
                    "top5_atfedes": top_k_overlap(a, b, k=5),
                    "top1_egyezes": top1_agreement(a, b),
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. mérés: SHAP vs LIME egyetértése
# ---------------------------------------------------------------------------

def measure_method_agreement(
    model_name: str,
    X_train: pd.DataFrame,
    cases: pd.DataFrame,
) -> pd.DataFrame:
    original_cols = list(X_train.columns)
    s_raw = shap_values(model_name, X_train, cases)
    l_raw = lime_values(model_name, X_train, cases)

    rows = []
    for i in range(len(cases)):
        a = aggregate_to_original(s_raw.iloc[i], original_cols)
        b = aggregate_to_original(l_raw.iloc[i], original_cols)
        rows.append({
            "modell": model_name,
            "eset": int(cases.index[i]),
            "spearman": rank_correlation(a, b),
            "top5_atfedes": top_k_overlap(a, b, k=5),
            "top1_egyezes": top1_agreement(a, b),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. mérés: a LIME belső zaja
# ---------------------------------------------------------------------------

def measure_lime_self_consistency(
    model_name: str,
    X_train: pd.DataFrame,
    cases: pd.DataFrame,
    n_runs: int = 3,
) -> pd.DataFrame:
    """Ugyanaz az eset, ugyanaz a modell, más véletlen mag.

    Ha ez alacsony, akkor a LIME bármely más eredményét óvatosan kell
    kezelni: a saját zaja elnyomhatja a valódi jelet.
    """
    original_cols = list(X_train.columns)
    runs = []
    for seed in range(n_runs):
        explainer = build_lime_explainer(model_name, X_train)
        explainer.random_state = np.random.RandomState(seed)
        runs.append(lime_values(model_name, X_train, cases, explainer=explainer))

    rows = []
    for i in range(len(cases)):
        for a_idx in range(n_runs):
            for b_idx in range(a_idx + 1, n_runs):
                a = aggregate_to_original(runs[a_idx].iloc[i], original_cols)
                b = aggregate_to_original(runs[b_idx].iloc[i], original_cols)
                rows.append({
                    "modell": model_name,
                    "eset": int(cases.index[i]),
                    "futas_par": f"{a_idx}-{b_idx}",
                    "spearman": rank_correlation(a, b),
                    "top5_atfedes": top_k_overlap(a, b, k=5),
                    "top1_egyezes": top1_agreement(a, b),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Ábra
# ---------------------------------------------------------------------------

def plot_demo(noise_df: pd.DataFrame, agree_df: pd.DataFrame,
              lime_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    # (a) zajstabilitás zajszintenként
    ax = axes[0]
    for method, color in [("shap", "#4C72B0"), ("lime", "#DD8452")]:
        sub = noise_df[noise_df["modszer"] == method]
        if sub.empty:
            continue
        grp = sub.groupby("zajszint")["top5_atfedes"]
        ax.errorbar(grp.mean().index * 100, grp.mean().values,
                    yerr=grp.std().values, marker="o", capsize=4,
                    color=color, label=method.upper())
    ax.set_xlabel("zajszint (a szórás %-ában)")
    ax.set_ylabel("top-5 átfedés az eredetivel")
    ax.set_ylim(0, 1.05)
    ax.set_title("(a) Zajstabilitás")
    ax.legend()
    ax.grid(alpha=0.3)

    # (b) SHAP vs LIME
    ax = axes[1]
    ax.hist(agree_df["top5_atfedes"], bins=np.arange(0, 1.2, 0.2),
            color="#55A868", edgecolor="white")
    ax.set_xlabel("SHAP-LIME top-5 átfedés")
    ax.set_ylabel("esetek száma")
    ax.set_title("(b) Módszerek egyetértése")
    ax.grid(alpha=0.3, axis="y")

    # (c) LIME önmagával
    ax = axes[2]
    ax.hist(lime_df["top5_atfedes"], bins=np.arange(0, 1.2, 0.2),
            color="#C44E52", edgecolor="white")
    ax.set_xlabel("LIME-LIME top-5 átfedés")
    ax.set_ylabel("összehasonlítások száma")
    ax.set_title("(c) A LIME belső zaja")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(f"Stabilitásmérés - demó {N_CASES_DEMO} eseten (XGBoost)")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "09_stability_demo.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------

def summarize(df: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    cols = ["spearman", "top5_atfedes", "top1_egyezes"]
    if group_cols:
        return df.groupby(group_cols)[cols].agg(["mean", "std"]).round(3)
    return df[cols].agg(["mean", "std"]).round(3)


def main() -> None:
    config.ensure_dirs()
    X_train, X_test, y_test = load_data()

    # Determinisztikus mintavétel a teszthalmazból
    rng = np.random.default_rng(config.RANDOM_STATE)
    idx = rng.choice(len(X_test), size=N_CASES_DEMO, replace=False)
    cases = X_test.iloc[np.sort(idx)]
    print(f"Vizsgált esetek: {list(cases.index)}\n")

    model = "xgboost"

    print("1/4  SHAP zajstabilitás...")
    shap_noise = measure_noise_stability(model, X_train, cases, method="shap")

    print("2/4  LIME zajstabilitás (lassú)...")
    lime_noise = measure_noise_stability(model, X_train, cases, method="lime")

    noise_df = pd.concat([shap_noise, lime_noise], ignore_index=True)

    print("3/4  SHAP vs LIME egyetértés...")
    agree_df = measure_method_agreement(model, X_train, cases)

    print("4/4  LIME önkonzisztencia...")
    lime_df = measure_lime_self_consistency(model, X_train, cases)

    # --- Eredmények ---
    print("\n=== 1. Zajstabilitás (eredeti vs zajosított magyarázat) ===")
    print(summarize(noise_df, ["modszer", "zajszint"]).to_string())

    print("\n=== 2. SHAP vs LIME egyetértése ===")
    print(summarize(agree_df).to_string())

    print("\n=== 3. LIME önmagával (más véletlen mag) ===")
    print(summarize(lime_df).to_string())

    out = config.PROJECT_ROOT / "reports" / "stability_demo.csv"
    noise_df.to_csv(out, index=False)
    agree_df.to_csv(out.with_name("agreement_demo.csv"), index=False)
    lime_df.to_csv(out.with_name("lime_selfconsistency_demo.csv"), index=False)

    plot_demo(noise_df, agree_df, lime_df)
    print(f"\nMentve: {out.name}, agreement_demo.csv, "
          f"lime_selfconsistency_demo.csv")
    print(f"Ábra: 09_stability_demo.png")


if __name__ == "__main__":
    main()
