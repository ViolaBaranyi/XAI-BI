"""Korrelációtudatos maszkoló hatása a magyarázatokra (kiegészítő kísérlet).

Futtatás:
    py -m src.masker_comparison               # 100 eset
    py -m src.masker_comparison --cases 30    # gyorsabb próba

MIÉRT KELL EZ?

A SHAP úgy méri egy jellemző hatását, hogy "elhagyja" azt, és a
háttéreloszlásból vett tipikus értékkel helyettesíti. Az alapértelmezett
`Independent` maszkoló a jellemzőket egymástól függetlennek tekinti, és
külön-külön cserélgeti őket.

Az adatkészletben viszont erős strukturális korreláció van: nyolc
szolgáltatásjellemző ugyanazon 1214 ügyfélnél veszi fel a "No internet
service" értéket. Ha ezeket függetlenül cserélgetjük, olyan ügyfelek
keletkeznek a háttérben, akik a valóságban nem létezhetnek - például
akinek nincs internete, de van online biztonsági csomagja. A modell
ezekre az irreális bemenetekre is ad jóslatot, és ez beépül a
SHAP-értékbe.

A `Partition` maszkoló a jellemzőket korreláció alapján csoportokba
rendezi, és csoportosan maszkolja őket. Ezzel az irreális kombinációk
nagy része elkerülhető.

MIT MÉRÜNK?

1. Mennyire tér el a két maszkoló magyarázata ugyanarra az ügyfélre.
2. Külön a korrelált szolgáltatáscsoportra: a csoport EGYÜTTES súlya
   marad-e, csak a csoporton belüli elosztás változik?

MÓDSZERTANI MEGJEGYZÉS: mindkét maszkolót ugyanazzal az algoritmussal
(shap.Explainer + valószínűségi kimenet) futtatjuk. Így a különbség
KIZÁRÓLAG a maszkolóból származik, nem az explainer típusából. Ezért nem
a korábbi TreeExplainer eredményeihez hasonlítunk: az log-odds skálán
dolgozik, ez pedig valószínűségi skálán.

Kimenet:
  reports/masker_comparison.csv
  reports/figures/13_masker_comparison.png
"""

import argparse
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import shap.maskers

from src import config
from src.explain import aggregate_to_original, load_data, load_pipeline
from src.stability import rank_correlation, top1_agreement, top_k_overlap

DEFAULT_CASES = 100
BACKGROUND_SIZE = 100

# Az a nyolc jellemző, amely ugyanannál az 1214 ügyfélnél veszi fel a
# "No internet service" értéket. Ezek alkotják a korrelált csoportot.
INTERNET_GROUP = [
    "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]


def build_explainers(model_name: str, X_train: pd.DataFrame):
    """Két magyarázó, azonos algoritmussal, csak a maszkolóban különböznek."""
    pipe = load_pipeline(model_name)
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    A = prep.transform(X_train)
    names = [f.split("__", 1)[1] for f in prep.get_feature_names_out()]

    def predict(x):
        return model.predict_proba(x)[:, 1]

    background = A[:BACKGROUND_SIZE]

    independent = shap.Explainer(
        predict,
        shap.maskers.Independent(background, max_samples=BACKGROUND_SIZE),
    )
    partition = shap.Explainer(
        predict,
        shap.maskers.Partition(background, clustering="correlation"),
    )
    return prep, names, {"independent": independent, "partition": partition}


def explain_all(explainer, prep, names, X_eval: pd.DataFrame) -> pd.DataFrame:
    values = explainer(prep.transform(X_eval), silent=True).values
    return pd.DataFrame(values, columns=names, index=X_eval.index)


def group_share(values: pd.Series, group: list[str]) -> float:
    """A korrelált csoport együttes súlya az összes hozzájárulásból."""
    total = values.abs().sum()
    return float(values[values.index.isin(group)].abs().sum() / total) if total else 0.0


def group_concentration(values: pd.Series, group: list[str]) -> float:
    """A csoporton belül mennyire egy jellemzőn összpontosul a súly.

    1,0 = minden a csoport egyetlen tagján; kis érték = szétoszlik.
    Ha a Partition összevonja a korrelált jellemzőket, ez nőhet.
    """
    inside = values[values.index.isin(group)].abs()
    return float(inside.max() / inside.sum()) if inside.sum() else 0.0


def plot_results(df: pd.DataFrame, global_imp: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    # (a) top-5 átfedés eloszlása
    ax = axes[0]
    ax.hist(df["top5_atfedes"], bins=np.arange(0, 1.2, 0.2),
            color="#4C72B0", edgecolor="white")
    ax.set_xlabel("top-5 átfedés a két maszkoló között")
    ax.set_ylabel("esetek száma")
    ax.set_title("(a) Mennyire egyeznek a magyarázatok?")
    ax.grid(alpha=0.3, axis="y")

    # (b) a korrelált csoport súlya
    ax = axes[1]
    ax.scatter(df["csoport_suly_ind"], df["csoport_suly_part"],
               alpha=0.5, color="#55A868", s=25)
    lim = [0, max(df[["csoport_suly_ind", "csoport_suly_part"]].max()) * 1.1]
    ax.plot(lim, lim, "k--", linewidth=1)
    ax.set_xlabel("Independent maszkoló")
    ax.set_ylabel("Partition maszkoló")
    ax.set_title("(b) Az internetszolgáltatás-csoport súlya")
    ax.grid(alpha=0.3)

    # (c) globális fontosság összevetése
    ax = axes[2]
    features = list(global_imp["independent"].head(10).index)
    x = np.arange(len(features))
    width = 0.38
    ax.barh(x - width / 2, global_imp["independent"][features].values,
            width, color="#4C72B0", label="Independent")
    ax.barh(x + width / 2, global_imp["partition"][features].values,
            width, color="#DD8452", label="Partition")
    ax.set_yticks(x)
    ax.set_yticklabels(features, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("átlagos |SHAP-érték|")
    ax.set_title("(c) Globális fontosság")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="x")

    fig.suptitle("Független vs korrelációtudatos maszkoló (XGBoost)")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "13_masker_comparison.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=DEFAULT_CASES)
    parser.add_argument("--model", default="xgboost")
    args = parser.parse_args()

    config.ensure_dirs()
    X_train, X_test, _ = load_data()

    rng = np.random.default_rng(config.RANDOM_STATE)
    idx = np.sort(rng.choice(len(X_test), size=args.cases, replace=False))
    cases = X_test.iloc[idx]
    original_cols = list(X_test.columns)

    print(f"Maszkoló-összehasonlítás: {args.cases} eset, modell: {args.model}\n")

    prep, names, explainers = build_explainers(args.model, X_train)

    raw = {}
    for key, explainer in explainers.items():
        t = time.time()
        raw[key] = explain_all(explainer, prep, names, cases)
        print(f"  {key:12s} {time.time() - t:5.1f} mp")

    # --- Esetenkénti összehasonlítás ---
    rows = []
    for i in range(len(cases)):
        a = aggregate_to_original(raw["independent"].iloc[i], original_cols)
        b = aggregate_to_original(raw["partition"].iloc[i], original_cols)
        rows.append({
            "eset": int(cases.index[i]),
            "spearman": rank_correlation(a, b),
            "top5_atfedes": top_k_overlap(a, b, k=5),
            "top1_egyezes": top1_agreement(a, b),
            "csoport_suly_ind": group_share(a, INTERNET_GROUP),
            "csoport_suly_part": group_share(b, INTERNET_GROUP),
            "csoport_koncentracio_ind": group_concentration(a, INTERNET_GROUP),
            "csoport_koncentracio_part": group_concentration(b, INTERNET_GROUP),
        })

    df = pd.DataFrame(rows)
    out = config.PROJECT_ROOT / "reports" / "masker_comparison.csv"
    df.to_csv(out, index=False)

    # --- Globális fontosság ---
    global_imp = {
        key: aggregate_to_original(m.abs().mean(), original_cols)
             .sort_values(ascending=False)
        for key, m in raw.items()
    }

    # --- Eredmények ---
    print("\n=== A két maszkoló egyetértése ===")
    print(df[["spearman", "top5_atfedes", "top1_egyezes"]]
          .agg(["mean", "std", "median"]).round(3).to_string())

    print("\n=== Globális top-5 ===")
    for key, imp in global_imp.items():
        print(f"  {key:12s}: {list(imp.head(5).index)}")

    print("\n=== A korrelált szolgáltatáscsoport kezelése ===")
    print(f"  Csoport együttes súlya   Independent: "
          f"{df['csoport_suly_ind'].mean():.3f} · Partition: "
          f"{df['csoport_suly_part'].mean():.3f}")
    print(f"  Csoporton belüli koncentráció  Independent: "
          f"{df['csoport_koncentracio_ind'].mean():.3f} · Partition: "
          f"{df['csoport_koncentracio_part'].mean():.3f}")

    shift = (df["csoport_suly_part"] - df["csoport_suly_ind"]).mean()
    direction = "nagyobb" if shift > 0 else "kisebb"
    print(f"\n  A korrelált csoport a Partition maszkolóval átlagosan "
          f"{abs(shift):.3f}-mal {direction} súlyt kap.")

    conc = (df["csoport_koncentracio_part"]
            - df["csoport_koncentracio_ind"]).mean()
    if conc > 0.02:
        print("  A Partition maszkoló a csoporton BELÜL koncentrálja a súlyt: "
              "kevesebb jellemző kapja a hozzájárulás nagyobb részét.")
    elif conc < -0.02:
        print("  A Partition maszkoló SZÉTOSZTJA a súlyt a csoport tagjai "
              "között.")
    else:
        print("  A csoporton belüli elosztás lényegében nem változik.")

    plot_results(df, global_imp)
    print(f"\nMentve: {out.name}")
    print("Ábra: 13_masker_comparison.png")


if __name__ == "__main__":
    main()
