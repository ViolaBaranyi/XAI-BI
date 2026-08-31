"""Modellek betanítása és értékelése (2. hét).

Futtatás:  py -m src.train

Három modellt tanítunk be szándékosan hasonló beállításokkal:
  1. Logisztikus regresszió - lineáris, önmagában is értelmezhető
  2. Döntési fa           - nemlineáris, de átlátható szabályokkal
  3. XGBoost              - "fekete doboz", a legpontosabb

FONTOS: nem célunk a maximális pontosság. A kutatási kérdés az, hogy
közel azonos teljesítményű modellek adnak-e eltérő magyarázatot
(Rashomon-hatás). Ha agyonhangolnánk a modelleket, pont ezt a
helyzetet rontanánk el.

Kimenet:
  models/*.joblib                    betanított pipeline-ok
  reports/metrics.csv                metrikatáblázat a dolgozathoz
  reports/figures/05_roc_curves.png  ROC-görbék
"""

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src import config
from src.prepare import build_preprocessor

# A modellek definíciója egy helyen. A kulcs lesz a fájlnév és a
# táblázat sorneve is, ezért rövid és beszédes.
MODELS = {
    "logreg": LogisticRegression(
        max_iter=1000,
        random_state=config.RANDOM_STATE,
    ),
    "tree": DecisionTreeClassifier(
        max_depth=5,               # korlátozva, hogy ne tanulja meg a zajt
        min_samples_leaf=20,
        random_state=config.RANDOM_STATE,
    ),
        "xgboost": XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=config.RANDOM_STATE,
        n_jobs=1,                  # egy szál: a párhuzamos összegzés nem determinisztikus
        tree_method="exact",       # a hisztogram-közelítés helyett pontos vágáskeresés
    ),
}

MODEL_LABELS = {
    "logreg": "Logisztikus regresszió",
    "tree": "Döntési fa",
    "xgboost": "XGBoost",
}


def load_data():
    """Betölti a 1. héten elkészített train/test halmazokat."""
    if not config.TRAIN_CSV.exists():
        raise FileNotFoundError("Futtasd előbb: py -m src.prepare")

    train_df = pd.read_csv(config.TRAIN_CSV)
    test_df = pd.read_csv(config.TEST_CSV)

    X_train = train_df.drop(columns=[config.TARGET])
    y_train = train_df[config.TARGET]
    X_test = test_df.drop(columns=[config.TARGET])
    y_test = test_df[config.TARGET]

    return X_train, y_train, X_test, y_test


def build_pipeline(name: str, X_train: pd.DataFrame) -> Pipeline:
    """Előfeldolgozó + modell egyetlen objektumban.

    Így a nyers (kódolatlan) adatot lehet beadni neki, ami a 6. héten a
    dashboardnál sokat fog érni. A 3. héten a SHAP a két lépést külön
    éri majd el: pipe.named_steps["prep"] és pipe.named_steps["model"].
    """
    return Pipeline([
        ("prep", build_preprocessor(X_train)),
        ("model", MODELS[name]),
    ])


def evaluate(pipe: Pipeline, X, y) -> dict:
    """A szokásos osztályozási metrikák egy halmazon."""
    y_pred = pipe.predict(X)
    y_proba = pipe.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_proba),
    }


def cross_validate(pipe: Pipeline, X, y) -> tuple[float, float]:
    """5-szörös rétegzett keresztvalidáció a tanulóhalmazon.

    Ez mutatja meg, mennyire stabil a modell teljesítménye - a szórás
    ugyanolyan fontos szám, mint az átlag.
    """
    cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=config.RANDOM_STATE
    )
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    return scores.mean(), scores.std()


def plot_roc_curves(results: dict, y_test) -> None:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    colors = {"logreg": "#4C72B0", "tree": "#DD8452", "xgboost": "#55A868"}

    for name, data in results.items():
        fpr, tpr, _ = roc_curve(y_test, data["y_proba"])
        ax.plot(fpr, tpr, color=colors[name], linewidth=2,
                label=f"{MODEL_LABELS[name]} (AUC = {data['test']['roc_auc']:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Véletlen találgatás")
    ax.set_xlabel("Fals pozitív arány")
    ax.set_ylabel("Valós pozitív arány")
    ax.set_title("ROC-görbék a teszthalmazon")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "05_roc_curves.png", dpi=150)
    plt.close(fig)


def main() -> None:
    config.ensure_dirs()

    X_train, y_train, X_test, y_test = load_data()
    print(f"Train: {X_train.shape[0]} sor, Test: {X_test.shape[0]} sor\n")

    results = {}
    rows = []

    for name in MODELS:
        print(f"--- {MODEL_LABELS[name]} ---")
        pipe = build_pipeline(name, X_train)

        cv_mean, cv_std = cross_validate(pipe, X_train, y_train)
        print(f"  CV ROC-AUC: {cv_mean:.4f} (+/- {cv_std:.4f})")

        pipe.fit(X_train, y_train)

        train_metrics = evaluate(pipe, X_train, y_train)
        test_metrics = evaluate(pipe, X_test, y_test)
        print(f"  Train ROC-AUC: {train_metrics['roc_auc']:.4f}")
        print(f"  Test  ROC-AUC: {test_metrics['roc_auc']:.4f}")

        # Túltanulás jelzése: ha a train sokkal jobb, mint a test
        gap = train_metrics["roc_auc"] - test_metrics["roc_auc"]
        if gap > 0.05:
            print(f"  FIGYELEM: {gap:.3f} a train-test különbség -> túltanulás")

        path = config.MODELS_DIR / f"{name}.joblib"
        joblib.dump(pipe, path)
        print(f"  Mentve: {path.name}\n")

        results[name] = {
            "pipe": pipe,
            "train": train_metrics,
            "test": test_metrics,
            "y_proba": pipe.predict_proba(X_test)[:, 1],
        }

        rows.append({
            "modell": MODEL_LABELS[name],
            "cv_roc_auc": round(cv_mean, 4),
            "cv_szoras": round(cv_std, 4),
            **{k: round(v, 4) for k, v in test_metrics.items()},
        })

    # --- Metrikatáblázat ---
    metrics_df = pd.DataFrame(rows)
    metrics_path = config.PROJECT_ROOT / "reports" / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print("=== Teszthalmaz eredményei ===")
    print(metrics_df.to_string(index=False))
    print(f"\nMentve: {metrics_path}")

    # --- ROC-görbék ---
    plot_roc_curves(results, y_test)
    print(f"Ábra: {config.FIGURES_DIR / '05_roc_curves.png'}")

    # --- A kutatási kérdés szempontjából lényeges ellenőrzés ---
    aucs = np.array([r["test"]["roc_auc"] for r in results.values()])
    spread = aucs.max() - aucs.min()
    print(f"\nA legjobb és leggyengébb modell AUC-különbsége: {spread:.4f}")
    if spread < 0.05:
        print("A három modell teljesítménye közel azonos. Pontosan ez kell:")
        print("így a magyarázatok eltérése nem a pontosságkülönbségből fakad.")


if __name__ == "__main__":
    main()
