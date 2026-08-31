"""SHAP és LIME magyarázatok egységes interfésszel (3. hét).

Futtatás:  py -m src.explain

A modul kulcsgondolata: a két magyarázó KÖZÖS formátumban ad eredményt
(egy pandas Series: jellemzőnév -> hozzájárulás). Enélkül a 4-5. héten
nem lehetne őket összemérni.

    magyarazat = explain_instance("xgboost", "shap", index=0)
    magyarazat = explain_instance("xgboost", "lime", index=0)

Két szint:
  - globális: mely jellemzők hajtják a modellt általában
  - lokális:  miért ezt jósolja EGY konkrét ügyfélre

Kimenet:
  reports/figures/06_shap_global.png       jellemzőfontosság három modellre
  reports/figures/07_shap_beeswarm.png     SHAP beeswarm (XGBoost)
  reports/figures/08_local_comparison.png  SHAP vs LIME egy ügyfélre
  reports/explanation_demo.csv             a fenti ügyfél számszerű adatai
"""

import warnings

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

from src import config

warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_LABELS = {
    "logreg": "Logisztikus regresszió",
    "tree": "Döntési fa",
    "xgboost": "XGBoost",
}


LIME_SAMPLES = 2000

# ---------------------------------------------------------------------------
# Betöltés és segédfüggvények
# ---------------------------------------------------------------------------

def load_pipeline(name: str):
    path = config.MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Nincs modell: {path}. Futtasd: py -m src.train")
    return joblib.load(path)


def load_data():
    train_df = pd.read_csv(config.TRAIN_CSV)
    test_df = pd.read_csv(config.TEST_CSV)
    X_train = train_df.drop(columns=[config.TARGET])
    X_test = test_df.drop(columns=[config.TARGET])
    y_test = test_df[config.TARGET]
    return X_train, X_test, y_test


def clean_names(preprocessor) -> list[str]:
    """A 'num__tenure' és 'cat__Contract_Two year' előtagok levágása."""
    return [f.split("__", 1)[1] for f in preprocessor.get_feature_names_out()]


def to_original_feature(encoded_name: str, original_columns: list[str]) -> str:
    """A one-hot oszlopot visszavezeti az eredeti jellemzőre.

    'Contract_Two year' -> 'Contract'

    Erre azért van szükség, mert a one-hot kódolás egy jellemzőt több
    oszlopra bont, és a hozzájárulása szétoszlik közöttük. Ha nem vonnánk
    össze, a SHAP és a LIME top-5 listája összehasonlíthatatlan lenne.
    """
    matches = [c for c in original_columns if encoded_name.startswith(c)]
    return max(matches, key=len) if matches else encoded_name


def aggregate_to_original(values: pd.Series, original_columns: list[str]) -> pd.Series:
    """Kódolt jellemzők hozzájárulásának összevonása eredeti jellemzőnként."""
    mapping = {c: to_original_feature(c, original_columns) for c in values.index}
    return values.groupby(mapping).sum()


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def shap_values(name: str, X_train: pd.DataFrame, X_eval: pd.DataFrame) -> pd.DataFrame:
    """SHAP-értékek mátrixa: sorok = esetek, oszlopok = kódolt jellemzők.

    A modell típusához illő magyarázót választjuk:
      - lineáris modellhez LinearExplainer (egzakt, gyors)
      - fa alapú modellhez TreeExplainer (egzakt, gyors)
    Modellfüggetlen KernelExplainer nem kell, mert mindkét családra van
    pontos megoldás. Ez fontos módszertani pont: a KernelExplainer
    közelít, tehát önmagában is instabil lenne.
    """
    pipe = load_pipeline(name)
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    A = prep.transform(X_train)
    B = prep.transform(X_eval)
    names = clean_names(prep)

    if name == "logreg":
        background = A[:1000]
        masker = shap.maskers.Independent(background, max_samples=background.shape[0])
        explainer = shap.LinearExplainer(model, masker)
    else:
        explainer = shap.TreeExplainer(model)

    values = np.array(explainer.shap_values(B))

    # A döntési fa mindkét osztályra ad értéket: (n, jellemzők, 2).
    # A pozitív osztály (lemorzsolódás) kell.
    if values.ndim == 3:
        values = values[:, :, 1]

    return pd.DataFrame(values, columns=names, index=X_eval.index)


# ---------------------------------------------------------------------------
# LIME
# ---------------------------------------------------------------------------

def build_lime_explainer(name: str, X_train: pd.DataFrame) -> LimeTabularExplainer:
    pipe = load_pipeline(name)
    prep = pipe.named_steps["prep"]
    return LimeTabularExplainer(
        prep.transform(X_train),
        feature_names=clean_names(prep),
        class_names=["marad", "lemorzsolódik"],
        mode="classification",
        random_state=config.RANDOM_STATE,
        discretize_continuous=True,
    )


def lime_values(
    name: str,
    X_train: pd.DataFrame,
    X_eval: pd.DataFrame,
    explainer: LimeTabularExplainer | None = None,
) -> pd.DataFrame:
    """LIME-értékek ugyanabban a formátumban, mint a SHAP-é.

    FIGYELEM: a LIME lassú, mert esetenként LIME_SAMPLES darab perturbált
    mintán tanít egy helyi lineáris modellt. Ne futtasd az egész
    teszthalmazra gondolkodás nélkül.
    """
    pipe = load_pipeline(name)
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    names = clean_names(prep)
    B = prep.transform(X_eval)

    if explainer is None:
        explainer = build_lime_explainer(name, X_train)

    rows = []
    for i in range(B.shape[0]):
        exp = explainer.explain_instance(
            B[i],
            model.predict_proba,
            num_features=len(names),
            num_samples=LIME_SAMPLES,
            labels=(1,),
        )
        row = np.zeros(len(names))
        for idx, weight in exp.as_map()[1]:
            row[idx] = weight
        rows.append(row)

    return pd.DataFrame(np.array(rows), columns=names, index=X_eval.index)


# ---------------------------------------------------------------------------
# Egységes belépési pont
# ---------------------------------------------------------------------------

def explain_instance(
    name: str,
    method: str,
    index: int,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    aggregate: bool = True,
) -> pd.Series:
    """EGY eset magyarázata, tetszőleges módszerrel.

    Ez az a függvény, amit a 4-5. héten a stabilitásmérés, a 6-7. héten
    pedig a dashboard fog hívni. A visszatérési érték mindig ugyanolyan
    alakú, függetlenül attól, hogy SHAP vagy LIME készítette.
    """
    row = X_test.iloc[[index]]

    if method == "shap":
        values = shap_values(name, X_train, row).iloc[0]
    elif method == "lime":
        values = lime_values(name, X_train, row).iloc[0]
    else:
        raise ValueError(f"Ismeretlen módszer: {method}")

    if aggregate:
        values = aggregate_to_original(values, list(X_test.columns))

    return values.sort_values(key=abs, ascending=False)


# ---------------------------------------------------------------------------
# Ábrák
# ---------------------------------------------------------------------------

def plot_global_importance(importances: dict, top_n: int = 12) -> None:
    """Globális jellemzőfontosság mindhárom modellre, egymás mellett."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharex=False)

    for ax, (name, imp) in zip(axes, importances.items()):
        top = imp.head(top_n).iloc[::-1]
        ax.barh(top.index, top.values, color="#4C72B0")
        ax.set_title(MODEL_LABELS[name], fontsize=11)
        ax.set_xlabel("átlagos |SHAP-érték|")
        ax.tick_params(axis="y", labelsize=9)

    fig.suptitle("Globális jellemzőfontosság (SHAP, eredeti jellemzőkre összevonva)")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "06_shap_global.png", dpi=150)
    plt.close(fig)


def plot_beeswarm(name: str, X_train: pd.DataFrame, X_eval: pd.DataFrame) -> None:
    """SHAP beeswarm: nemcsak azt mutatja, mennyire fontos egy jellemző,
    hanem azt is, hogy magas vagy alacsony értéke tolja-e a jóslatot."""
    pipe = load_pipeline(name)
    prep = pipe.named_steps["prep"]
    B = prep.transform(X_eval)
    sv = shap_values(name, X_train, X_eval)

    plt.figure()
    shap.summary_plot(
        sv.values, B, feature_names=list(sv.columns), max_display=15, show=False
    )
    plt.title(f"SHAP beeswarm - {MODEL_LABELS[name]}", fontsize=11)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "07_shap_beeswarm.png", dpi=150)
    plt.close()


def plot_local_comparison(shap_exp: pd.Series, lime_exp: pd.Series,
                          case_id: int, top_n: int = 8) -> None:
    """Ugyanaz az ügyfél, két magyarázó. Ez a hét fő ábrája."""
    features = list(dict.fromkeys(
        list(shap_exp.head(top_n).index) + list(lime_exp.head(top_n).index)
    ))
    s = shap_exp.reindex(features).fillna(0)
    l = lime_exp.reindex(features).fillna(0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, vals, title in [(axes[0], s, "SHAP"), (axes[1], l, "LIME")]:
        colors = ["#C44E52" if v > 0 else "#4C72B0" for v in vals.values]
        ax.barh(vals.index[::-1], vals.values[::-1], color=colors[::-1])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("hozzájárulás")
        ax.tick_params(axis="y", labelsize=9)

    fig.suptitle(f"Ugyanazon ügyfél magyarázata két módszerrel "
                 f"(teszthalmaz #{case_id}, piros = lemorzsolódás felé tol)")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "08_local_comparison.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------

def top_k_overlap(a: pd.Series, b: pd.Series, k: int = 5) -> float:
    """Hány százalékban egyezik a két magyarázó top-k listája.

    Ez a 4. hét egyik metrikájának előfutára.
    """
    set_a = set(a.head(k).index)
    set_b = set(b.head(k).index)
    return len(set_a & set_b) / k


def main() -> None:
    config.ensure_dirs()
    X_train, X_test, y_test = load_data()

    # A globális elemzéshez elég egy részminta - a SHAP így is gyors,
    # de a beeswarm 1400 ponttal már olvashatatlan lenne.
    sample = X_test.head(300)

    print("=== Globális magyarázatok (SHAP) ===")
    importances = {}
    for name in ["logreg", "tree", "xgboost"]:
        sv = shap_values(name, X_train, sample)
        agg = aggregate_to_original(sv.abs().mean(), list(X_test.columns))
        importances[name] = agg.sort_values(ascending=False)
        top5 = list(importances[name].head(5).index)
        print(f"  {MODEL_LABELS[name]:24s} top5: {top5}")

    plot_global_importance(importances)
    plot_beeswarm("xgboost", X_train, sample)
    print(f"\nÁbrák: 06_shap_global.png, 07_shap_beeswarm.png")

    # Modellek közti egyetértés globálisan
    print("\n=== Modellek egyetértése (globális top-5) ===")
    pairs = [("logreg", "tree"), ("logreg", "xgboost"), ("tree", "xgboost")]
    for a, b in pairs:
        ov = top_k_overlap(importances[a], importances[b], k=5)
        print(f"  {a:8s} vs {b:8s}: {ov:.0%}")

    # --- Lokális magyarázat egy konkrét ügyfélre ---
    # Olyat választunk, akit a modell magas kockázatúnak tart: ott van tétje
    # a magyarázatnak.
    pipe = load_pipeline("xgboost")
    proba = pipe.predict_proba(X_test)[:, 1]
    case_id = int(np.argsort(proba)[-10])   # a 10. legkockázatosabb ügyfél

    print(f"\n=== Lokális magyarázat: teszthalmaz #{case_id} ===")
    print(f"  Jósolt lemorzsolódási valószínűség: {proba[case_id]:.3f}")
    print(f"  Valós címke: {y_test.iloc[case_id]}")

    s_exp = explain_instance("xgboost", "shap", case_id, X_train, X_test)
    print("  SHAP top5:", list(s_exp.head(5).index))

    print("  LIME futtatása (lassú, ~10 mp)...")
    l_exp = explain_instance("xgboost", "lime", case_id, X_train, X_test)
    print("  LIME top5:", list(l_exp.head(5).index))

    overlap = top_k_overlap(s_exp, l_exp, k=5)
    print(f"\n  SHAP-LIME top-5 átfedés ezen az eseten: {overlap:.0%}")

    plot_local_comparison(s_exp, l_exp, case_id)
    print("  Ábra: 08_local_comparison.png")

    demo = pd.DataFrame({"shap": s_exp, "lime": l_exp.reindex(s_exp.index)})
    demo_path = config.PROJECT_ROOT / "reports" / "explanation_demo.csv"
    demo.to_csv(demo_path)
    print(f"  Mentve: {demo_path.name}")


if __name__ == "__main__":
    main()
