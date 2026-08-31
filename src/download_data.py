"""Az adatkészlet letöltése.

Futtatás:  python -m src.download_data

Az adatot NEM commitoljuk a repóba (lásd .gitignore) - helyette ez a
szkript teszi reprodukálhatóvá a letöltést.
"""

import urllib.request

import pandas as pd

from src import config


def download(force: bool = False) -> None:
    config.ensure_dirs()

    if config.RAW_CSV.exists() and not force:
        print(f"Az adat már létezik: {config.RAW_CSV}")
    else:
        print(f"Letöltés innen: {config.DATA_URL}")
        urllib.request.urlretrieve(config.DATA_URL, config.RAW_CSV)
        print(f"Elmentve ide: {config.RAW_CSV}")

    df = pd.read_csv(config.RAW_CSV)
    print(f"\nMéret: {df.shape[0]} sor x {df.shape[1]} oszlop")
    print(f"Célváltozó eloszlása:\n{df[config.TARGET].value_counts()}")


if __name__ == "__main__":
    download()
