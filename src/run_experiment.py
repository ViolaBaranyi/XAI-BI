"""A teljes kísérlet lefuttatása (5. hét).

Futtatás:
    py -m src.run_experiment              # 100 eset, mindhárom modell
    py -m src.run_experiment --cases 50   # gyorsabb próba
    py -m src.run_experiment --models xgboost

A 4. hét demója 10 eseten futott, ami anekdota. Itt ugyanaz fut le
100+ eseten, mindhárom modellre, és statisztikai értékelést is kap:
bootstrap konfidenciaintervallumot és Wilcoxon-próbát.

FIGYELEM: a LIME lassú. 100 eset x 3 modell kb. 20-45 perc, géptől
függően. Egyszer kell lefuttatni, aztán az eredmény CSV-ből dolgozol.

Kimenet:
  reports/experiment_noise.csv        zajstabilitás, minden mérés
  reports/experiment_agreement.csv    SHAP vs LIME
  reports/experiment_lime_self.csv    LIME önkonzisztencia
  reports/experiment_summary.csv      összefoglaló táblázat a dolgozathoz
  reports/figures/10_noise_stability.png
  reports/figures/11_method_agreement.png
  reports/figures/12_top1_risk.png
"""

import argparse
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src import config
from src.explain import MODEL_LABELS, load_data
from src.stability import (
    NOISE_LEVELS,
    measure_lime_self_consistency,
    measure_method_agreement,
    measure_noise_stability,
)

DEFAULT_CASES = 100
N_BOOTSTRAP = 2000


# ---------------------------------------------------------------------------
# Statisztika
# ---------------------------------------------------------------------------

def bootstrap_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """95%-os bootstrap konfidenciaintervallum az átlagra.

    Azért bootstrap és nem t-próba: a top-5 átfedés diszkrét értékeket
    vesz fel (0; 0,2; 0,4; ...), az eloszlása erősen nem normális.
    """
    values = np.asarray(values)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return (np.nan, np.nan)

    rng = np.random.default_rng(config.RANDOM_STATE)
    means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(N_BOOTSTRAP)
    ])
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def describe(values: np.ndarray, label: str) -> dict:
    values = np.asarray(values)
    values = values[~np.isnan(values)]
    lo, hi = bootstrap_ci(values)
    return {
        "metrika": label,
        "n": len(values),
        "atlag": round(float(values.mean()), 3),
        "szoras": round(float(values.std(ddof=1)), 3),
        "median": round(float(np.median(values)), 3),
        "ci_also": round(lo, 3),
        "ci_felso": round(hi, 3),
    }


def paired_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Wilcoxon előjeles rangpróba két párosított minta között.

    Nem t-próbát használunk, mert az adat nem normális eloszlású, és
    ugyanazokon az eseteken mérünk kétféle módszert - tehát párosított.
    """
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = np.asarray(a)[mask], np.asarray(b)[mask]
    if len(a) < 10 or np.allclose(a, b):
        return (np.nan, np.nan)
    stat, p = wilcoxon(a, b)
    return (float(stat), float(p))


# ---------------------------------------------------------------------------
# Ábrák
# ---------------------------------------------------------------------------

COLORS = {"logreg": "#4C72B0", "tree": "#DD8452", "xgboost": "#55A868"}


def plot_noise_stability(noise: pd.DataFrame) -> None:
    """Zajstabilitás modellenként és módszerenként."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, method in zip(axes, ["shap", "lime"]):
        sub = noise[noise["modszer"] == method]
        for model in sub["modell"].unique():
            grp = sub[sub["modell"] == model].groupby("zajszint")["top5_atfedes"]
            x = grp.mean().index * 100
            y = grp.mean().values
            err = grp.sem().values * 1.96
            ax.errorbar(x, y, yerr=err, marker="o", capsize=4,
                        color=COLORS.get(model, "gray"),
                        label=MODEL_LABELS.get(model, model))
        ax.set_xlabel("zajszint (a szórás %-ában)")
        ax.set_title(method.upper())
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("top-5 átfedés az eredeti magyarázattal")
    axes[0].legend(fontsize=9)
    fig.suptitle("A magyarázatok zajstabilitása (95%-os konfidenciasávval)")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "10_noise_stability.png", dpi=150)
    plt.close(fig)


def plot_method_agreement(agree: pd.DataFrame, lime_self: pd.DataFrame) -> None:
    """SHAP-LIME egyetértés, a LIME saját zajszintjéhez viszonyítva."""
    fig, ax = plt.subplots(figsize=(8, 5))

    models = list(agree["modell"].unique())
    x = np.arange(len(models))
    width = 0.35

    a_means = [agree[agree["modell"] == m]["top5_atfedes"].mean() for m in models]
    a_err = [agree[agree["modell"] == m]["top5_atfedes"].sem() * 1.96 for m in models]
    l_means = [lime_self[lime_self["modell"] == m]["top5_atfedes"].mean() for m in models]
    l_err = [lime_self[lime_self["modell"] == m]["top5_atfedes"].sem() * 1.96 for m in models]

    ax.bar(x - width / 2, a_means, width, yerr=a_err, capsize=4,
           color="#55A868", label="SHAP vs LIME")
    ax.bar(x + width / 2, l_means, width, yerr=l_err, capsize=4,
           color="#C44E52", label="LIME vs önmaga (zajküszöb)")

    ax.set_xticks(x, [MODEL_LABELS.get(m, m) for m in models], fontsize=9)
    ax.set_ylabel("top-5 átfedés")
    ax.set_ylim(0, 1.05)
    ax.set_title("Két magyarázó egyetértése a LIME saját zajszintjéhez képest")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "11_method_agreement.png", dpi=150)
    plt.close(fig)


def plot_top1_risk(noise: pd.DataFrame) -> None:
    """Az üzletileg legfontosabb ábra: milyen arányban változik meg a
    legfontosabb tényező a zaj hatására."""
    fig, ax = plt.subplots(figsize=(8, 5))

    shap_noise = noise[noise["modszer"] == "shap"]
    for model in shap_noise["modell"].unique():
        grp = shap_noise[shap_noise["modell"] == model].groupby("zajszint")
        risk = 1 - grp["top1_egyezes"].mean()
        ax.plot(risk.index * 100, risk.values, marker="o",
                color=COLORS.get(model, "gray"),
                label=MODEL_LABELS.get(model, model))

    ax.set_xlabel("zajszint (a szórás %-ában)")
    ax.set_ylabel("a fő tényező megváltozásának aránya")
    ax.set_title("Mekkora eséllyel mutat mást a dashboard fő indoklása? (SHAP)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "12_top1_risk.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=DEFAULT_CASES)
    parser.add_argument("--models", nargs="+",
                        default=["logreg", "tree", "xgboost"])
    args = parser.parse_args()

    config.ensure_dirs()
    X_train, X_test, y_test = load_data()

    rng = np.random.default_rng(config.RANDOM_STATE)
    idx = np.sort(rng.choice(len(X_test), size=args.cases, replace=False))
    cases = X_test.iloc[idx]

    print(f"Kísérlet: {args.cases} eset, modellek: {args.models}")
    print(f"Zajszintek: {[f'{n:.0%}' for n in NOISE_LEVELS]}\n")

    all_noise, all_agree, all_self = [], [], []
    t0 = time.time()

    for model in args.models:
        print(f"--- {MODEL_LABELS.get(model, model)} ---")

        print("  SHAP zajstabilitás...", end="", flush=True)
        t = time.time()
        all_noise.append(measure_noise_stability(model, X_train, cases, "shap"))
        print(f" {time.time() - t:.0f} mp")

        print("  LIME zajstabilitás...", end="", flush=True)
        t = time.time()
        all_noise.append(measure_noise_stability(model, X_train, cases, "lime"))
        print(f" {time.time() - t:.0f} mp")

        print("  SHAP vs LIME...", end="", flush=True)
        t = time.time()
        all_agree.append(measure_method_agreement(model, X_train, cases))
        print(f" {time.time() - t:.0f} mp")

        print("  LIME önkonzisztencia...", end="", flush=True)
        t = time.time()
        all_self.append(measure_lime_self_consistency(model, X_train, cases))
        print(f" {time.time() - t:.0f} mp\n")

    noise = pd.concat(all_noise, ignore_index=True)
    agree = pd.concat(all_agree, ignore_index=True)
    lime_self = pd.concat(all_self, ignore_index=True)

    reports = config.PROJECT_ROOT / "reports"
    noise.to_csv(reports / "experiment_noise.csv", index=False)
    agree.to_csv(reports / "experiment_agreement.csv", index=False)
    lime_self.to_csv(reports / "experiment_lime_self.csv", index=False)

    print(f"Teljes futásidő: {(time.time() - t0) / 60:.1f} perc\n")

    # --- Összefoglaló táblázat ---
    rows = []
    for model in args.models:
        for method in ["shap", "lime"]:
            for sigma in NOISE_LEVELS:
                sub = noise[(noise["modell"] == model) &
                            (noise["modszer"] == method) &
                            (noise["zajszint"] == sigma)]
                d = describe(sub["top5_atfedes"].values,
                             f"zaj {sigma:.0%} top5")
                d.update({"modell": model, "modszer": method})
                rows.append(d)

        sub = agree[agree["modell"] == model]
        d = describe(sub["top5_atfedes"].values, "SHAP vs LIME top5")
        d.update({"modell": model, "modszer": "shap-lime"})
        rows.append(d)

        sub = lime_self[lime_self["modell"] == model]
        d = describe(sub["top5_atfedes"].values, "LIME zajküszöb top5")
        d.update({"modell": model, "modszer": "lime-lime"})
        rows.append(d)

    summary = pd.DataFrame(rows)[
        ["modell", "modszer", "metrika", "n", "atlag", "szoras",
         "median", "ci_also", "ci_felso"]
    ]
    summary.to_csv(reports / "experiment_summary.csv", index=False)

    print("=== Összefoglaló (top-5 átfedés) ===")
    print(summary.to_string(index=False))

    # --- Statisztikai próbák ---
    print("\n=== Wilcoxon-próba: SHAP vs LIME zajstabilitása ===")
    print("(páronként, ugyanazokon az eseteken)")
    for model in args.models:
        for sigma in NOISE_LEVELS:
            s = noise[(noise["modell"] == model) & (noise["modszer"] == "shap") &
                      (noise["zajszint"] == sigma)].sort_values(["eset", "ismetles"])
            l = noise[(noise["modell"] == model) & (noise["modszer"] == "lime") &
                      (noise["zajszint"] == sigma)].sort_values(["eset", "ismetles"])
            stat, p = paired_test(s["top5_atfedes"].values, l["top5_atfedes"].values)
            sig = "szignifikáns" if p < 0.05 else "nem szignifikáns"
            print(f"  {model:8s} zaj={sigma:.0%}: p={p:.4f} ({sig})")

    print("\n=== A LIME zajválasza megkülönböztethető-e a saját zajától? ===")
    for model in args.models:
        threshold = lime_self[lime_self["modell"] == model]["top5_atfedes"].mean()
        for sigma in NOISE_LEVELS:
            val = noise[(noise["modell"] == model) &
                        (noise["modszer"] == "lime") &
                        (noise["zajszint"] == sigma)]["top5_atfedes"].mean()
            diff = val - threshold
            if abs(diff) < 0.05:
                verdict = "NEM - a zajküszöb hibahatárán belül"
            elif diff > 0:
                verdict = "NEM - a zaj hatása kisebb, mint a módszer saját ingadozása"
            else:
                verdict = "igen"
            print(f"  {model:8s} zaj={sigma:.0%}: "
                  f"zajstabilitás={val:.3f}, zajküszöb={threshold:.3f}, "
                  f"eltérés={diff:+.3f} -> {verdict}")

    # --- Ábrák ---
    plot_noise_stability(noise)
    plot_method_agreement(agree, lime_self)
    plot_top1_risk(noise)
    print("\nÁbrák: 10_noise_stability.png, 11_method_agreement.png, "
          "12_top1_risk.png")


if __name__ == "__main__":
    main()
