# XAI a Business Intelligence döntéstámogatásban

Kutatási prototípus: mennyire megbízhatók a SHAP és LIME magyarázatok egy
üzleti döntéstámogató (BI) környezetben?

**Kutatási kérdés.** Közel azonos pontosságú modellek eltérő magyarázatot
adhatnak ugyanarra az esetre, és a magyarázatok kis adatzajra is
megváltozhatnak. Ez a projekt megméri, mennyire stabilak a post hoc
magyarázatok egy lemorzsolódás-előrejelzési feladaton, és egy dashboardon
megjeleníti a magyarázat mellé annak megbízhatóságát is.

- **Adat:** IBM Telco Customer Churn (7043 ügyfél, 19 jellemző)
- **Modellek:** logisztikus regresszió, döntési fa, XGBoost
- **Magyarázók:** SHAP, LIME

## Telepítés

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Futtatás

```bash
python -m src.download_data     # adat letöltése a data/raw/ mappába
python -m src.prepare           # tisztítás + train/test bontás
python -m src.eda               # feltáró elemzés, ábrák a reports/figures/ mappába
```

## Mappastruktúra

```
src/
  config.py          minden útvonal és beállítás (fix seed!)
  download_data.py   adatletöltés
  prepare.py         tisztítás, bontás, előfeldolgozó pipeline
  eda.py             feltáró elemzés
data/
  raw/               nyers CSV (nincs verziókövetve)
  processed/         train.csv, test.csv (nincs verziókövetve)
models/              betanított modellek 
reports/figures/     ábrák a dolgozathoz
NAPLO.md             
```

## Ütemterv

| Hét | Feladat | Állapot |
|-----|---------|---------|
| 1 | Repo, adat, EDA, train/test bontás | ✅ |
| 2 | Három modell betanítása, metrikák | ✅ |
| 3 | SHAP és LIME integrálása | ⬜ |
| 4 | Stabilitás- és egyetértés-metrika megtervezése | ⬜ |
| 5 | Kísérlet futtatása, eredmények | ⬜ |
| 6 | Streamlit dashboard váz | ⬜ |
| 7 | Dashboard: modellváltó + stabilitásjelzés | ⬜ |
| 8 | Szakirodalmi fejezet megírása | ⬜ |
| 9 | Módszertan és eredmények megírása | ⬜ |
| 10 | Lezárás, prezentáció, puffer | ⬜ |
