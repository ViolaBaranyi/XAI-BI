# Kutatási napló

> Ezt hetente vezesd. A 8-9. héten, amikor írni kell, ez lesz a
> módszertani fejezet nyersanyaga - utólag nem fogsz emlékezni rá,
> miért döntöttél úgy, ahogy.

Minden hétnél három dolgot rögzíts: **mit csináltam**, **milyen döntést
hoztam és miért**, **mi akadt el**.

---

## 1. hét

**Mit csináltam:** repo létrehozása, IBM Telco Churn adat letöltése,
tisztítás, rétegzett 80/20 bontás, feltáró elemzés.

**Döntések:**

- *Adatkészlet:* Telco Churn. Indok: tabuláris, beszédes jellemzőnevek
  (a SHAP ábrákon ez fontos), közepes méret (7043 sor - a SHAP gyorsan
  lefut rajta), és a szakirodalomban is gyakori, tehát összehasonlítható.
- *TotalCharges hiányzó értékek:* 11 sorban üres string. Ezeknél a tenure
  értéke 0, tehát új ügyfelekről van szó -> 0-val pótoltam, nem dobtam el.
- *Kódolás nem itt történik:* a one-hot és a skálázás a modell
  pipeline-jába kerül (2. hét), hogy a mentett CSV olvasható maradjon és
  ne legyen adatszivárgás a train/test között.
- *Fix seed (42):* enélkül a stabilitásmérés eredménye nem reprodukálható.

**Megfigyelések az EDA-ból:**

- 26,4% a lemorzsolódási arány - enyhén kiegyensúlyozatlan, de nem
  drasztikusan. Az 5. héten meg lehet nézni, mi történik SMOTE-tal.
- Legerősebb jelek: `Contract` (kéthavi szerződésnél 2,9%, havinál 42,7%),
  `PaymentMethod` = Electronic check (45,2%), `TechSupport` hiánya.
- Nyolc jellemzőben szerepel a "No internet service" érték ugyanarra az
  1214 ügyfélre. Ez erős multikollinearitás, és a magyarázatok
  szempontjából érdekes: a SHAP szét fogja osztani a hatást ezek között.
  **Ezt érdemes megjegyezni - lehet, hogy ez lesz az egyik eredményed.**

**Elakadás:** -

---

## 2. hét

**Mit csináltam:** három modell betanítása (logisztikus regresszió, döntési fa,
XGBoost) egységes pipeline-ban, 5-szörös rétegzett keresztvalidáció,
metrikatáblázat és ROC-görbék.

**Döntések:**

- *Nem hangoltam a hiperparamétereket.* A kutatási kérdés nem a pontosság,
  hanem a magyarázatok stabilitása. Ha az egyik modell jóval pontosabb lenne,
  a magyarázatok eltérését a pontosságkülönbségnek lehetne betudni - így
  viszont nem lehet.
- *Döntési fa max_depth=5, min_samples_leaf=20.* Korlátozás nélkül a fa
  memorizálná a tanulóhalmazt, és a magyarázata értelmetlen lenne.
- *Pipeline-t használok, nem külön előfeldolgozót.* Így a keresztvalidáció
  minden hajtásban újratanítja a skálázót, tehát nincs adatszivárgás.
- *Nem kezeltem a 26%-os osztálykiegyensúlyozatlanságot.* Az 5. hétre marad
  annak vizsgálata, hogy a SMOTE hogyan hat a magyarázatok stabilitására
  (a CIES-cikk épp ezt találta érdekesnek).

**Eredmények:**

| Modell | CV ROC-AUC | Teszt ROC-AUC | Teszt F1 |
|---|---|---|---|
| Logisztikus regresszió | 0,846 (±0,016) | 0,840 | 0,583 |
| Döntési fa | 0,829 (±0,014) | 0,828 | 0,597 |
| XGBoost | 0,838 (±0,018) | 0,837 | 0,573 |

**Megfigyelések:**

- A három modell teszt-AUC-ja között mindössze **0,012** a különbség.
  Ez a kutatás szempontjából ideális kiindulópont: gyakorlatilag azonos
  teljesítményű modellek, tehát ha eltérő magyarázatot adnak, az a
  Rashomon-hatás és nem a képességkülönbség.
- Az XGBoost train-test AUC különbsége 0,086 - enyhén túltanul, de a
  teszteredménye így is versenyképes. Erős regularizációval csökkenthető,
  de akkor eltérnék a "ne hangoljunk" elvtől. A dolgozatban ezt jelezni kell.
- A recall mindhárom modellnél 0,52-0,57 körül van: a lemorzsolódók
  csaknem felét nem találjuk meg. Üzletileg ez fontos korlát, és a
  küszöbérték hangolásával javítható lenne.

**Elakadás:** -

---

## 3. hét

**Mit csináltam:** SHAP és LIME integrálása egységes interfészen keresztül,
globális és lokális magyarázatok mindhárom modellre, első összehasonlítás.

**Döntések:**

- *Modellspecifikus SHAP-magyarázó.* Lineáris modellhez LinearExplainer,
  fa alapúhoz TreeExplainer. Mindkettő egzakt megoldást ad. A modellfüggetlen
  KernelExplainer közelít, tehát önmagában is instabil - ha azt használnám,
  nem tudnám elkülöníteni a módszer zaját a modell tulajdonságaitól.
- *One-hot oszlopok visszavonása eredeti jellemzőkre.* A kódolás egy
  jellemzőt több oszlopra bont, a hozzájárulás szétoszlik. Az összevonás
  nélkül a SHAP és a LIME top-k listája összehasonlíthatatlan lenne.
  Az aggregálás a hozzájárulások összege (a Shapley-értékek additívak,
  tehát ez elméletileg is megalapozott).
- *LIME_SAMPLES = 2000.* A LIME esetenként ennyi perturbált mintán tanít
  helyi lineáris modellt. Kisebb szám gyorsabb, de instabilabb - az 5. héten
  ezt külön megvizsgálom.
- *Közös visszatérési formátum (pandas Series).* A stabilitásmérés (4-5. hét)
  és a dashboard (6-7. hét) ugyanazt a függvényt fogja hívni.

**Eredmények:**

Globális top-5 jellemzők (SHAP, 300 elemű részmintán):

| Modell | Top-5 |
|---|---|
| Logisztikus regresszió | tenure, Contract, InternetService, TotalCharges, MonthlyCharges |
| Döntési fa | Contract, InternetService, tenure, MonthlyCharges, OnlineSecurity |
| XGBoost | Contract, tenure, MonthlyCharges, TotalCharges, InternetService |

Modellek közti globális top-5 átfedés: logreg-xgboost 100%, logreg-fa 80%,
fa-xgboost 80%.

Lokális magyarázat (teszthalmaz #1161, jósolt kockázat 0,912, valós címke 1):

- SHAP top-5: TotalCharges, tenure, Contract, MonthlyCharges, InternetService
- LIME top-5: Contract, tenure, InternetService, OnlineSecurity, PaymentMethod
- **Átfedés: 60%**

**Megfigyelések - ez a hét legfontosabb tanulsága:**

- Globálisan a három modell nagyrészt egyetért (80-100%), de a SORREND
  eltér: a logisztikus regressziónál a tenure vezet, a másik kettőnél a
  Contract. Ha egy BI dashboard csak a legfontosabb tényezőt mutatná,
  modellválasztástól függően MÁS választ adna ugyanarra a kérdésre.
- Lokálisan sokkal rosszabb az egyezés: ugyanarra az ügyfélre, UGYANARRA
  a modellre a SHAP és a LIME top-5 listája csak 60%-ban fedi egymást.
  A SHAP szerint a TotalCharges a legerősebb tényező, a LIME szerint a
  Contract. Egy üzleti felhasználó a kettőből teljesen más beavatkozást
  vezetne le.
- A hozzájárulások nagyságrendje is más (SHAP: log-odds skála, LIME: helyi
  lineáris együttható), tehát az ABSZOLÚT értékek nem hasonlíthatók össze,
  csak a sorrend. Ezt a dolgozatban tisztázni kell.

**Elakadás:** -

---

## 4. hét

**Mit csináltam:** a mérési módszertan megtervezése és implementálása,
demófuttatás 10 eseten (XGBoost).

**A mérés felépítése:**

Három külön kérdést mérünk, mert ezeket el kell tudni különíteni egymástól:

1. *Zajstabilitás* - a numerikus jellemzőkre Gauss-zajt teszünk a jellemző
   szórásának 1/5/10%-ában, és az eredeti magyarázathoz hasonlítjuk.
2. *Módszerek egyetértése* - ugyanaz a modell és eset, SHAP vs LIME.
3. *A LIME önkonzisztenciája* - ugyanaz minden, csak a véletlen mag más.

A harmadik a kontrollcsoport szerepét tölti be: e nélkül nem lehetne
megmondani, hogy a LIME zajra adott válasza valódi érzékenység-e, vagy
csak a módszer saját ingadozása.

**Döntések:**

- *Zaj a szórás arányában, nem abszolút értékben.* A tenure 0-72 hónap,
  a TotalCharges 0-8600 dollár. Abszolút zaj az egyiket szétverné, a
  másikat meg sem karcolná.
- *Csak a folytonos jellemzőket zajosítom* (tenure, MonthlyCharges,
  TotalCharges). A SeniorCitizen bináris, azon a "kis zaj" értelmetlen.
- *Két metrika párhuzamosan.* A Spearman-rangkorreláció a teljes sorrendet
  nézi, a top-5 átfedés azt, amit a felhasználó a képernyőn lát. Külön
  mérem a top-1 egyezést is: egy dashboard fejlécében a legfontosabb okot
  emelnénk ki, tehát annak ingadozása közvetlen üzleti kockázat.
- *Abszolút érték szerinti rangsor.* A felhasználót az érdekli, mely
  tényezők fontosak, nem az, hogy melyik irányba tolnak.
- *Minden zajszinthez rögzített véletlen mag* (RANDOM_STATE + ismétlés),
  hogy a mérés újrafuttatható legyen.

**Demó eredmények (10 eset, XGBoost, 3 ismétlés zajszintenként):**

Zajstabilitás - top-5 átfedés az eredeti magyarázattal:

| Módszer | 1% zaj | 5% zaj | 10% zaj |
|---|---|---|---|
| SHAP | 0,927 | 0,860 | 0,840 |
| LIME | 0,813 | 0,760 | 0,773 |

Top-1 egyezés (ugyanaz maradt-e a legfontosabb tényező):

| Módszer | 1% zaj | 5% zaj | 10% zaj |
|---|---|---|---|
| SHAP | 0,900 | 0,767 | 0,733 |
| LIME | 0,900 | 0,867 | 0,933 |

SHAP vs LIME egyetértése: Spearman 0,552; top-5 átfedés 0,620; top-1 0,700.

LIME önkonzisztenciája: Spearman 0,803; top-5 átfedés 0,767; top-1 0,933.

**Megfigyelések - ez a demó máris két erős állítást ad:**

1. **A SHAP érzékenyebbnek LÁTSZIK a zajra, mint a LIME, de ez félrevezető.**
   A LIME zajra adott top-5 átfedése (0,76-0,81) gyakorlatilag megegyezik a
   saját, zaj nélküli önkonzisztenciájával (0,767). Vagyis a LIME válasza a
   zajra nem különböztethető meg a módszer belső véletlenétől: annyira
   ingadozik magától, hogy nem tudja kimutatni a bemenet változását.
   A SHAP determinisztikus, tehát nála a 0,84-0,93 tiszta jel.
   **Ez az eredmény csak azért látszik, mert külön mértem a LIME belső
   zaját.** E nélkül épp fordított következtetésre jutottam volna.

2. **10%-os adatzajnál a SHAP top-1 tényezője az esetek 27%-ában megváltozik.**
   Ez konkrét üzleti kockázat: egy dashboard, amely a "fő okot" kiemeli,
   négyből egy ügyfélnél mást írna ki egy reális adatminőségi ingadozás
   mellett.

3. A SHAP és a LIME top-5 listája átlagosan csak 62%-ban fedi egymást,
   a teljes sorrend korrelációja 0,55. Ugyanaz a modell, ugyanaz az
   ügyfél - a magyarázó megválasztása érdemben megváltoztatja a választ.

**Futásidő:** 10 eset = 38 másodperc. 100 esetre kb. 6-7 perc, ami az
5. héten bőven vállalható.

**Elakadás:** -

---

## 5. hét

**Mit csináltam:** a teljes kísérlet lefuttatása mindhárom modellre,
statisztikai értékeléssel (bootstrap CI, Wilcoxon-próba) és a dolgozat
fő ábráival.

**Döntések:**

- *Bootstrap konfidenciaintervallum t-próba helyett.* A top-5 átfedés
  diszkrét értékeket vesz fel (0; 0,2; 0,4; 0,6; 0,8; 1,0), az eloszlása
  erősen nem normális, tehát a normalitást feltételező eljárások nem
  megfelelőek.
- *Wilcoxon előjeles rangpróba* a SHAP és a LIME összehasonlítására:
  párosított minta (ugyanazokon az eseteken mérünk kétféle módszert),
  nem normális eloszlás.
- *A LIME önkonzisztenciája mint ZAJKÜSZÖB.* Ez a kísérlet módszertani
  gerince: minden LIME-eredményt ehhez viszonyítok. Ha a zajra adott
  válasz nem tér el a küszöbtől, akkor a mérés nem mutat ki semmit.

**Eredmények (25 eset, próbafuttatás - a végleges 100 esetes fut még):**

Zajstabilitás, top-5 átfedés:

| Modell | Módszer | 1% | 5% | 10% |
|---|---|---|---|---|
| Logisztikus regresszió | SHAP | 0,979 | 0,981 | 0,976 |
| Döntési fa | SHAP | 1,000 | 0,987 | 0,960 |
| XGBoost | SHAP | 0,947 | 0,864 | 0,824 |
| Logisztikus regresszió | LIME | 0,821 | 0,832 | 0,816 |
| Döntési fa | LIME | 0,845 | 0,845 | 0,832 |
| XGBoost | LIME | 0,808 | 0,829 | 0,808 |

SHAP vs LIME top-5 átfedés: logreg 0,760; döntési fa 0,640; XGBoost 0,704.

LIME zajküszöb (önkonzisztencia): logreg 0,827; fa 0,813; XGBoost 0,789.

**Fő megállapítások:**

1. **A LIME zajválasza egyetlen modellnél, egyetlen zajszinten sem
   különböztethető meg a saját zajküszöbétől.** A LIME görbéje mindhárom
   modellnél vízszintes: 1% és 10% zaj között gyakorlatilag nincs
   különbség (0,808-0,845 tartomány), és ez egybeesik a saját
   önkonzisztenciájával (0,789-0,827). A LIME tehát ezen a feladaton
   nem alkalmas stabilitásvizsgálatra: a saját mintavételi zaja
   elnyomja a bemenet változásának hatását.

2. **A modell komplexitása erősen befolyásolja a SHAP stabilitását.**
   A korlátozott mélységű döntési fánál 1% zajnál tökéletes (1,000) az
   egyezés, mert a durva vágások nem érzékenyek kis elmozdulásra.
   Az XGBoost-nál 10% zajnál már csak 0,824. Ugyanaz a magyarázó,
   ugyanaz az adat - a stabilitás a modelltől függ, nem csak a
   módszertől. Ez fontos üzenet a BI gyakorlat felé: a legpontosabb
   modell választása egyben a legingatagabb magyarázatot is jelentheti.

3. **A Wilcoxon-próba az XGBoost-nál 5% és 10% zajnál NEM szignifikáns**
   (p=0,067 és p=0,694), miközben minden más esetben p<0,001. Ennek oka,
   hogy a SHAP stabilitása itt lecsökken a LIME szintjére. Vagyis a
   legösszetettebb modellnél elvész a SHAP előnye.

4. A két módszer egyetértése a döntési fánál a legrosszabb (0,640),
   pedig az a legegyszerűbb modell. Ez ellentmond annak az intuíciónak,
   hogy egyszerűbb modellt könnyebb magyarázni.

**Futásidő:** 25 eset x 3 modell = 76 másodperc. 100 esetre kb. 5-15 perc.

**Elakadás:** -

---

## 6. hét

**Mit csináltam:**

**Döntések:**

**Elakadás:**
