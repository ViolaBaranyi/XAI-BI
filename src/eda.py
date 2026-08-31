"""Feltáró adatelemzés (EDA).

Futtatás:  python -m src.eda

Kimenet: ábrák a reports/figures/ mappában + szöveges összegzés a konzolon.
Ezek az ábrák a szakdolgozat "Adatok" alfejezetébe kerülnek majd.
"""

import matplotlib

matplotlib.use("Agg")  # fájlba mentés, megjelenítő ablak nélkül

import matplotlib.pyplot as plt
import pandas as pd

from src import config
from src.prepare import get_feature_names


def summary(df: pd.DataFrame) -> None:
    print("--- Alapstatisztikák ---")
    numeric, categorical = get_feature_names(df)
    print(df[numeric].describe().round(2).to_string())

    print("\n--- Hiányzó értékek ---")
    missing = df.isna().sum()
    print("Nincs hiányzó érték." if missing.sum() == 0
          else missing[missing > 0].to_string())

    print("\n--- Kategorikus jellemzők egyedi értékei ---")
    for col in categorical:
        vals = df[col].unique()
        print(f"  {col:20s} ({len(vals)}): {list(vals)}")

    print("\n--- Lemorzsolódási arány kategóriánként (top eltérések) ---")
    base = df[config.TARGET].mean()
    rows = []
    for col in categorical:
        for val, grp in df.groupby(col):
            rows.append((col, str(val), len(grp), grp[config.TARGET].mean()))
    tbl = pd.DataFrame(rows, columns=["jellemző", "érték", "n", "churn_arány"])
    tbl["eltérés"] = (tbl["churn_arány"] - base).abs()
    print(f"Átlagos lemorzsolódás: {base:.3f}")
    print(tbl.sort_values("eltérés", ascending=False).head(10)
          .round(3).to_string(index=False))


def plot_target_balance(df: pd.DataFrame) -> None:
    counts = df[config.TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(["Maradt (0)", "Lemorzsolódott (1)"], counts.values,
           color=["#4C72B0", "#C44E52"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v}\n({v / len(df):.1%})", ha="center", va="bottom")
    ax.set_title("Célváltozó eloszlása")
    ax.set_ylabel("Ügyfelek száma")
    ax.set_ylim(0, counts.max() * 1.2)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "01_target_balance.png", dpi=150)
    plt.close(fig)


def plot_numeric_distributions(df: pd.DataFrame) -> None:
    cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col in zip(axes, cols):
        for label, color, name in [(0, "#4C72B0", "Maradt"),
                                   (1, "#C44E52", "Lemorzsolódott")]:
            ax.hist(df.loc[df[config.TARGET] == label, col], bins=30,
                    alpha=0.6, color=color, label=name, density=True)
        ax.set_title(col)
        ax.set_xlabel(col)
    axes[0].set_ylabel("sűrűség")
    axes[0].legend()
    fig.suptitle("Numerikus jellemzők eloszlása lemorzsolódás szerint")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "02_numeric_distributions.png", dpi=150)
    plt.close(fig)


def plot_churn_by_category(df: pd.DataFrame) -> None:
    """A négy legerősebb kategorikus jellemző lemorzsolódási aránya."""
    cols = ["Contract", "InternetService", "PaymentMethod", "TechSupport"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    base = df[config.TARGET].mean()
    for ax, col in zip(axes.ravel(), cols):
        rates = df.groupby(col)[config.TARGET].mean().sort_values()
        ax.barh(rates.index.astype(str), rates.values, color="#55A868")
        ax.axvline(base, color="black", linestyle="--", linewidth=1,
                   label=f"átlag ({base:.2f})")
        ax.set_title(col)
        ax.set_xlabel("lemorzsolódási arány")
        ax.legend(fontsize=8)
    fig.suptitle("Lemorzsolódási arány kategóriánként")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "03_churn_by_category.png", dpi=150)
    plt.close(fig)


def plot_correlation(df: pd.DataFrame) -> None:
    numeric, _ = get_feature_names(df)
    corr = df[numeric + [config.TARGET]].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Numerikus jellemzők korrelációja")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "04_correlation.png", dpi=150)
    plt.close(fig)


def main() -> None:
    config.ensure_dirs()
    if not config.TRAIN_CSV.exists():
        raise FileNotFoundError(
            "Nincs meg a tanulóhalmaz. Futtasd előbb: python -m src.prepare"
        )

    df = pd.read_csv(config.TRAIN_CSV)
    print(f"Tanulóhalmaz: {df.shape[0]} sor x {df.shape[1]} oszlop\n")

    summary(df)

    plot_target_balance(df)
    plot_numeric_distributions(df)
    plot_churn_by_category(df)
    plot_correlation(df)

    print(f"\nÁbrák elmentve ide: {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
