"""Adattisztítás és train/test bontás.

Futtatás:  python -m src.prepare

Kimenet: data/processed/train.csv és test.csv

Fontos döntés: a kódolást (one-hot, skálázás) NEM itt végezzük el, hanem
a modell pipeline-jában (2. hét). Így a mentett CSV-k emberi szemmel is
olvashatók maradnak, ami a magyarázatok ellenőrzésénél (3-5. hét) sokat ér.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def load_raw() -> pd.DataFrame:
    if not config.RAW_CSV.exists():
        raise FileNotFoundError(
            "Nincs meg a nyers adat. Futtasd előbb: python -m src.download_data"
        )
    return pd.read_csv(config.RAW_CSV)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Tisztítás. Minden lépést kommentálunk, mert ez a szakdolgozat
    módszertani fejezetének a nyersanyaga."""
    df = df.copy()

    # 1. A TotalCharges szövegként érkezik, és 11 sorban üres string szerepel
    #    (ezek az ügyfelek 0 hónapja vannak a szolgáltatónál).
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_missing = df["TotalCharges"].isna().sum()
    if n_missing:
        print(f"  TotalCharges: {n_missing} hiányzó érték -> 0-val pótolva "
              f"(tenure == 0 esetek)")
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # 2. Az azonosító nem prediktor, eldobjuk.
    df = df.drop(columns=[config.ID_COLUMN])

    # 3. A célváltozó Yes/No -> 1/0.
    df[config.TARGET] = (df[config.TARGET] == "Yes").astype(int)

    # 4. Duplikátumok kiszűrése.
    n_dup = df.duplicated().sum()
    if n_dup:
        print(f"  {n_dup} duplikált sor eltávolítva")
        df = df.drop_duplicates()

    return df.reset_index(drop=True)


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rétegzett bontás: a lemorzsolódási arány a train és test halmazban
    is ugyanaz maradjon."""
    train_df, test_df = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=df[config.TARGET],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def get_feature_names(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Visszaadja a numerikus és kategorikus jellemzők nevét."""
    features = [c for c in df.columns if c != config.TARGET]
    numeric = [c for c in features if c in config.NUMERIC_FEATURES]
    categorical = [c for c in features if c not in config.NUMERIC_FEATURES]
    return numeric, categorical


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Előfeldolgozó a 2. héthez: numerikus skálázás + one-hot kódolás.

    A `handle_unknown="ignore"` azért kell, hogy a dashboard (6. hét) ne
    haljon el egy ismeretlen kategóriaértéken.
    """
    numeric, categorical = get_feature_names(df)
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop=None), categorical),
        ],
        remainder="drop",
    )


def main() -> None:
    config.ensure_dirs()

    print("Nyers adat betöltése...")
    df = load_raw()
    print(f"  {df.shape[0]} sor x {df.shape[1]} oszlop")

    print("Tisztítás...")
    df = clean(df)

    print("Bontás...")
    train_df, test_df = split(df)

    train_df.to_csv(config.TRAIN_CSV, index=False)
    test_df.to_csv(config.TEST_CSV, index=False)

    numeric, categorical = get_feature_names(df)
    rate_tr = train_df[config.TARGET].mean()
    rate_te = test_df[config.TARGET].mean()

    print("\n--- Összegzés ---")
    print(f"Train: {len(train_df)} sor, lemorzsolódási arány {rate_tr:.3f}")
    print(f"Test:  {len(test_df)} sor, lemorzsolódási arány {rate_te:.3f}")
    print(f"Numerikus jellemzők ({len(numeric)}): {numeric}")
    print(f"Kategorikus jellemzők ({len(categorical)}): {categorical}")
    print(f"\nMentve: {config.TRAIN_CSV}")
    print(f"Mentve: {config.TEST_CSV}")


if __name__ == "__main__":
    main()
