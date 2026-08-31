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

**Mit csináltam:**

**Döntések:**

**Elakadás:**
