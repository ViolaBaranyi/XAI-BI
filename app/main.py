"""Ügyfélmegtartási döntéstámogató - felhasználóbarát változat (7. hét).

Futtatás a projekt gyökeréből:
    py -m streamlit run app/main.py

A korábbi változat kutatói szemmel készült: azonosítók, nyers százalékok,
szakkifejezések. Ez a verzió ugyanazt tudja, de úgy, hogy egy
ügyfélszolgálati munkatárs is használni tudja:

  - beszédes ügyfélnevek a puszta sorszám helyett
  - minden szám mellett rövid, magyar nyelvű magyarázat
  - a legfontosabb üzenet mondatban is megjelenik, nemcsak ábrán
  - beépített útmutató és fogalomtár
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src import config
from src.explain import (
    MODEL_LABELS,
    aggregate_to_original,
    lime_values,
    load_pipeline,
    shap_values,
)
from src.stability import PERTURB_FEATURES, perturb, top_k_overlap

st.set_page_config(page_title="Ügyfélmegtartás", layout="wide",
                   initial_sidebar_state="expanded")

TOP_N = 8
STABILITY_SIGMA = 0.05
STABILITY_REPS = 8
LIME_NOISE_FLOOR = 0.86

# Magyar nevek a jellemzőknek. A nyers oszlopnevek angolul vannak, és egy
# üzleti felhasználónak semmit nem mondanak.
FEATURE_LABELS = {
    "tenure": "Ügyfélkapcsolat hossza",
    "Contract": "Szerződés típusa",
    "MonthlyCharges": "Havi díj",
    "TotalCharges": "Eddig elköltött összeg",
    "InternetService": "Internetszolgáltatás",
    "OnlineSecurity": "Online biztonsági csomag",
    "OnlineBackup": "Online mentés",
    "TechSupport": "Technikai támogatás",
    "DeviceProtection": "Eszközvédelem",
    "StreamingTV": "TV-streaming",
    "StreamingMovies": "Film-streaming",
    "PaymentMethod": "Fizetési mód",
    "PaperlessBilling": "Papírmentes számlázás",
    "PhoneService": "Telefonszolgáltatás",
    "MultipleLines": "Több vonal",
    "SeniorCitizen": "Nyugdíjas korú",
    "Partner": "Van párja",
    "Dependents": "Eltartottak",
    "gender": "Nem",
}

CONTRACT_HU = {
    "Month-to-month": "havi",
    "One year": "1 éves",
    "Two year": "2 éves",
}


def label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature)


# ---------------------------------------------------------------------------
# Betöltés
# ---------------------------------------------------------------------------

@st.cache_data
def load_frames():
    return pd.read_csv(config.TRAIN_CSV), pd.read_csv(config.TEST_CSV)


@st.cache_resource
def get_pipeline(name: str):
    return load_pipeline(name)


@st.cache_data
def get_predictions(name: str) -> pd.Series:
    _, test_df = load_frames()
    X = test_df.drop(columns=[config.TARGET])
    return pd.Series(get_pipeline(name).predict_proba(X)[:, 1], index=X.index)


def _explain_rows(name: str, method: str, X_train, rows) -> pd.DataFrame:
    if method == "SHAP":
        return shap_values(name, X_train, rows)
    return lime_values(name, X_train, rows)


@st.cache_data
def get_explanation(name: str, method: str, case_id: int) -> pd.Series:
    train_df, test_df = load_frames()
    X_train = train_df.drop(columns=[config.TARGET])
    X_test = test_df.drop(columns=[config.TARGET])
    raw = _explain_rows(name, method, X_train, X_test.iloc[[case_id]]).iloc[0]
    agg = aggregate_to_original(raw, list(X_test.columns))
    return agg.sort_values(key=abs, ascending=False)


@st.cache_data
def get_stability(name: str, method: str, case_id: int) -> dict:
    train_df, test_df = load_frames()
    X_train = train_df.drop(columns=[config.TARGET])
    X_test = test_df.drop(columns=[config.TARGET])
    cols = list(X_test.columns)
    stds = X_train[PERTURB_FEATURES].std()

    base = get_explanation(name, method, case_id)
    base_top1 = base.abs().idxmax()

    scores, top1_hits = [], []
    counter: dict[str, int] = {}

    for rep in range(STABILITY_REPS):
        rng = np.random.default_rng(config.RANDOM_STATE + rep)
        noisy = perturb(X_test.iloc[[case_id]], STABILITY_SIGMA, stds, rng)
        agg = aggregate_to_original(
            _explain_rows(name, method, X_train, noisy).iloc[0], cols)
        scores.append(top_k_overlap(base, agg, k=5))
        top1_hits.append(int(agg.abs().idxmax() == base_top1))
        for f in agg.abs().sort_values(ascending=False).head(5).index:
            counter[f] = counter.get(f, 0) + 1

    return {
        "score": float(np.mean(scores)),
        "top1_rate": float(np.mean(top1_hits)),
        "frequency": {f: c / STABILITY_REPS for f, c in counter.items()},
    }


# ---------------------------------------------------------------------------
# Szöveges értékelés
# ---------------------------------------------------------------------------

def risk_sentence(risk: float) -> tuple[str, str]:
    if risk > 0.75:
        return ("error", "Nagyon valószínű, hogy ez az ügyfél felmond. "
                         "Érdemes minél előbb megkeresni.")
    if risk > 0.5:
        return ("warning", "Az ügyfél a kockázatos sávba esik. "
                           "Megtartó ajánlat megfontolandó.")
    if risk > 0.25:
        return ("info", "Mérsékelt kockázat. Egyelőre nem sürgős, de figyelendő.")
    return ("success", "Alacsony kockázat. Nem igényel beavatkozást.")


def stability_sentence(stab: dict, method: str, top1: str) -> tuple[str, str]:
    score, rate = stab["score"], stab["top1_rate"]

    if method == "LIME":
        return ("warning",
                "A LIME módszer válasza ezen az adaton nem megbízható: "
                "ugyanarra az ügyfélre futásonként mást mond. "
                "A megbízhatósági jelzéshez válts SHAP-ra a bal oldalon.")

    if rate < 0.5:
        return ("error",
                f"**Ne hivatkozz egyetlen fő okra.** Ha az ügyfél adatai "
                f"csak kicsit is pontatlanok, a lista élére más tényező "
                f"kerül – az esetek {1 - rate:.0%}-ában így történt. "
                f"A(z) „{label(top1)}” tehát nem kiemelhető egyedüli okként.")

    if score >= 0.9 and rate >= 0.9:
        return ("success",
                f"**Megbízható indoklás.** A fő ok – „{label(top1)}” – "
                f"akkor is ugyanaz marad, ha az adatok kissé pontatlanok. "
                f"Nyugodtan hivatkozhatsz rá az ügyféllel folytatott "
                f"beszélgetésben.")

    if score >= 0.75:
        return ("warning",
                f"**Nagyjából megbízható.** A legfontosabb tényezők stabilak, "
                f"de a lista alsó fele változékony. A halvány oszlopokra ne "
                f"építs érvelést.")

    return ("error",
            "**Ingatag indoklás.** Kis adatpontatlanság is átrendezi a "
            "sorrendet. Használd tájékozódásra, de ne érvelj vele.")


def direction_word(value: float) -> str:
    return "növeli" if value > 0 else "csökkenti"


# ---------------------------------------------------------------------------
# Ábra
# ---------------------------------------------------------------------------

def explanation_chart(values: pd.Series, frequency: dict, top_n: int = TOP_N):
    top = values.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(top) + 1))
    for i, (name, v) in enumerate(top.items()):
        freq = frequency.get(name, 0.0)
        alpha = 1.0 if freq >= 0.75 else (0.5 if freq >= 0.5 else 0.25)
        ax.barh(i, v, color="#C44E52" if v > 0 else "#4C72B0", alpha=alpha)

    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([label(f) for f in top.index], fontsize=10)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("← kisebb kockázat        nagyobb kockázat →", fontsize=9)
    ax.set_xticks([])
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------

def select_customer():
    st.title("Ügyfélmegtartás – kockázatelemzés")
    st.caption(
        "Ez az eszköz megbecsüli, mely ügyfelek készülnek felmondani, "
        "megmutatja, mire alapozza a becslést, és jelzi, mennyire "
        "megbízható ez az indoklás."
    )

    train_df, test_df = load_frames()
    X_test = test_df.drop(columns=[config.TARGET])
    y_test = test_df[config.TARGET]

    # --- Oldalsáv ---
    # A visszaállítás úgy működik, hogy a widgetek ÚJ kulcsot kapnak.
    # A session_state kulcsainak törlése nem elég: a böngésző visszaküldi
    # a régi értéket. Új kulcs = a widgetnek nincs mentett állapota, tehát
    # a kódban megadott alapértelmezéssel épül fel.
    nonce = st.session_state.get("filter_nonce", 0)

    with st.sidebar:
        st.header("Szűrők")

        model = st.selectbox(
            "Előrejelző modell",
            options=["xgboost", "logreg", "tree"],
            key=f"f_model_{nonce}",
            format_func=lambda m: MODEL_LABELS[m],
            help="Három különböző modell, közel azonos pontossággal. "
                 "Váltogasd őket ugyanazon az ügyfélen: látni fogod, hogy "
                 "más-más indoklást adnak.",
        )

        proba = get_predictions(model)

        risk_min, risk_max = st.slider(
            "Felmondás esélye (%)", 0, 100, (50, 100), step=5,
            key=f"f_risk_{nonce}",
            help="Csak az ebbe a sávba eső ügyfelek jelennek meg.",
        )

        contracts = st.multiselect(
            "Szerződés típusa",
            options=list(CONTRACT_HU.keys()),
            default=list(CONTRACT_HU.keys()),
            format_func=lambda c: CONTRACT_HU[c],
            key=f"f_contract_{nonce}",
        )

        max_tenure = int(X_test["tenure"].max())
        tenure_range = st.slider(
            "Ügyfélkapcsolat hossza (hónap)", 0, max_tenure, (0, max_tenure),
            key=f"f_tenure_{nonce}",
        )

        st.divider()
        method = st.radio(
            "Indoklás módszere", ["SHAP", "LIME"], horizontal=True,
            key=f"f_method_{nonce}",
            help="Két elterjedt módszer arra, hogy egy modell döntését "
                 "utólag megmagyarázzuk. A mérések szerint a SHAP "
                 "megbízhatóbb ezen az adaton.",
        )

        if st.button("Szűrők alaphelyzetbe", width="stretch"):
            st.session_state["filter_nonce"] = nonce + 1
            st.rerun()

    # --- Szűrés ---
    mask = (
        (proba >= risk_min / 100) & (proba <= risk_max / 100)
        & X_test["Contract"].isin(contracts if contracts else CONTRACT_HU)
        & X_test["tenure"].between(*tenure_range)
    )
    pool = proba[mask].sort_values(ascending=False)

    if pool.empty:
        st.warning("Nincs a szűrőknek megfelelő ügyfél. "
                   "Tágítsd a beállításokat a bal oldalon.")
        st.stop()

    # --- Útmutató ---
    with st.expander("Hogyan használd? (kattints ide)", expanded=False):
        st.markdown(
            """
**1. lépés.** A táblázatból kattints egy ügyfélre. A bal oldali
szűrőkkel leszűkítheted a listát – kockázat, szerződéstípus vagy
ügyfélkapcsolat hossza szerint. Az oszlopfejlécre kattintva rendezhetsz,
a táblázat fölé húzva az egeret pedig kereshetsz is.

**2. lépés.** Nézd meg a *kockázatot*: ez azt mutatja, a modell szerint
mekkora eséllyel mond fel az ügyfél a közeljövőben.

**3. lépés.** Az *indoklásból* látod, mely tényezők tolják felfelé vagy
lefelé ezt a kockázatot. A piros sávok növelik, a kékek csökkentik.

**4. lépés – ez a legfontosabb.** Nézd meg a *megbízhatósági jelzést*
az indoklás mellett. Ha azt írja, hogy megbízható, hivatkozhatsz az okra
az ügyféllel folytatott beszélgetésben. Ha azt írja, hogy ingatag, akkor
a rendszer nem tudja biztosan, mi a valódi ok – ilyenkor ne mondj
konkrét indokot az ügyfélnek.

**Miért van erre szükség?** Ezek a magyarázatok utólagos becslések a
modell működéséről. Ha az ügyfél adatai csak kicsit pontatlanok –
márpedig a gyakorlatban azok –, az indoklás átrendeződhet. A rendszer
ezt leméri, és szól, ha ez a helyzet.
            """
        )

    # --- Ügyfélválasztó táblázat ---
    st.subheader("Válassz ügyfelet")
    st.caption(
        f"**{len(pool)}** ügyfél felel meg a szűrőknek "
        f"(átlagos kockázat: {pool.mean():.0%}). "
        "Kattints egy sorra az elemzéshez. Az oszlopfejlécekre kattintva "
        "rendezhetsz, a nagyító ikonnal kereshetsz."
    )

    listing = pd.DataFrame({
        "Kockázat": (pool.values * 100).round(0),
        "Hónap": X_test.loc[pool.index, "tenure"].values,
        "Szerződés": [CONTRACT_HU.get(c, c)
                      for c in X_test.loc[pool.index, "Contract"]],
        "Havi díj": X_test.loc[pool.index, "MonthlyCharges"].values.round(0),
        "Internet": X_test.loc[pool.index, "InternetService"].values,
    })

    event = st.dataframe(
        listing,
        hide_index=True,
        width="stretch",
        height=280,
        key=f"customer_table_{nonce}",
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Kockázat": st.column_config.ProgressColumn(
                "Felmondás esélye", format="%.0f%%", min_value=0, max_value=100,
            ),
            "Hónap": st.column_config.NumberColumn(
                "Ügyfélkapcsolat", format="%d hó",
            ),
            "Havi díj": st.column_config.NumberColumn(format="$%d"),
        },
    )

    selected_rows = event.selection.rows if event and event.selection else []
    position = selected_rows[0] if selected_rows else 0
    case_id = int(pool.index[position])

    if not selected_rows:
        st.info("A lista legkockázatosabb ügyfelét mutatjuk. "
                "Kattints egy másik sorra a váltáshoz.")

    st.divider()

    return model, method, case_id, proba, X_test, y_test


def render_customer_tab(model, method, case_id, proba, X_test, y_test):
    """A kiválasztott ügyfél részletes elemzése."""
    # --- Kockázat ---
    risk = proba[case_id]
    actual = y_test.iloc[case_id]
    row = X_test.iloc[case_id]

    st.subheader("A kiválasztott ügyfél kockázata")
    c1, c2 = st.columns([1, 3])
    with c1:
        st.metric("Felmondás esélye", f"{risk:.0%}",
                  help="A modell becslése arra, hogy az ügyfél a "
                       "közeljövőben felmondja a szerződését.")
    with c2:
        st.progress(float(risk))
        level, text = risk_sentence(risk)
        getattr(st, level)(text)

    st.divider()

    with st.spinner("Indoklás és megbízhatóság kiszámítása..."):
        values = get_explanation(model, method, case_id)
        stab = get_stability(model, method, case_id)

    top1 = values.abs().idxmax()
    top1_value = values[top1]

    # --- Indoklás ---
    st.subheader("Miért gondolja ezt a rendszer?")

    st.markdown(
        f"A legerősebb tényező: **{label(top1)}** "
        f"(értéke: `{row[top1]}`), amely {direction_word(top1_value)} "
        f"a felmondás kockázatát."
    )

    left, right = st.columns([3, 2])

    with left:
        st.pyplot(explanation_chart(values, stab["frequency"]))
        st.caption(
            "🔴 Piros: növeli a kockázatot · 🔵 Kék: csökkenti · "
            "**A halvány sávok bizonytalanok** – ezek a tényezők eltűnnek a "
            "listáról, ha az adat kissé pontatlan. A jobb oldali táblázat "
            "megmondja, melyikre hivatkozhatsz."
        )

    with right:
        st.markdown("**Mennyire bízhatsz ebben az indoklásban?**")
        level, text = stability_sentence(stab, method, top1)
        getattr(st, level)(text)

        if method == "SHAP":
            m1, m2 = st.columns(2)
            m1.metric(
                "Indoklás stabilitása", f"{stab['score']:.0%}",
                help="Ha az ügyfél adatait apró mértékben megváltoztatjuk, "
                     "a fontos tényezők ekkora része marad ugyanaz. "
                     "90% felett megbízható.",
            )
            m2.metric(
                "Fő ok megmarad", f"{stab['top1_rate']:.0%}",
                help="Ennyiszer maradt ugyanaz a legfontosabb tényező. "
                     "Ha ez alacsony, ne hivatkozz egyetlen okra.",
            )

        st.markdown("**Mely tényezőkre hivatkozhatsz?**")
        rows = []
        for f in values.head(TOP_N).index:
            freq = stab["frequency"].get(f, 0.0)
            if freq >= 0.75:
                mark = "✅ BIZTOS OK"
            elif freq >= 0.5:
                mark = "⚠️ BIZONYTALAN"
            else:
                mark = "⛔ NE HIVATKOZZ RÁ"
            rows.append({
                "Tényező": label(f),
                "Ítélet": mark,
                "Megbízhatóság": round(freq * 100),
            })
        st.dataframe(
            pd.DataFrame(rows), hide_index=True, width="stretch",
            column_config={
                "Megbízhatóság": st.column_config.ProgressColumn(
                    "Megbízhatóság", format="%d%%", min_value=0, max_value=100,
                    help="Ennyiszer maradt benne a legfontosabb öt tényező "
                         "között, amikor az adatot apró mértékben "
                         "megváltoztattuk.",
                ),
            },
        )
        st.caption(
            "✅ **Biztos ok** – nyugodtan említheted az ügyfélnek. "
            "⚠️ **Bizonytalan** – csak óvatosan. "
            "⛔ **Ne hivatkozz rá** – a rendszer nem tudja biztosan, hogy "
            "ez tényleg számít-e."
        )

    st.divider()

    # --- Részletek ---
    col_a, col_b = st.columns(2)

    with col_a:
        with st.expander("Az ügyfél összes adata"):
            # Az Érték oszlop szöveget és számot is tartalmaz; Arrow ezt
            # nem tudja egyetlen típusra hozni, ezért mindent szöveggé
            # alakítunk a megjelenítéshez.
            display = pd.DataFrame({
                "Adat": [label(c) for c in row.index],
                "Érték": [str(v) for v in row.values],
            })
            st.dataframe(display, hide_index=True, width="stretch",
                         height=400)

    with col_b:
        with st.expander("Fogalomtár"):
            st.markdown(
                f"""
**Felmondás esélye** – a modell becslése 0 és 100% között. Nem jóslat:
azt fejezi ki, mennyire hasonlít ez az ügyfél azokra, akik korábban
felmondtak.

**Indoklás** – utólagos becslés arról, mely adatok mozdították a modell
döntését. Nem ok-okozati összefüggés: azt írja le, hogyan működik a
modell, nem azt, hogy mi történik a valóságban.

**Indoklás stabilitása** – az ügyfél adatait {STABILITY_REPS} alkalommal
apró mértékben ({STABILITY_SIGMA:.0%}) megváltoztatjuk, és megnézzük,
ugyanazt az indoklást kapjuk-e. Minél magasabb, annál inkább
támaszkodhatsz rá.

**SHAP és LIME** – két elterjedt magyarázó módszer. A 200 ügyfélen végzett
mérés szerint a LIME ezen az adaton futásonként mást mond ugyanarra az
ügyfélre, ezért a megbízhatósági jelzés csak SHAP-nál értelmezhető.

**Modell** – a kiválasztott előrejelző eljárás. Három modell szerepel,
gyakorlatilag azonos pontossággal, de eltérő indoklásokkal. Ez nem hiba:
ugyanaz a jelenség többféleképpen is leírható.
                """
            )

        with st.expander("Ellenőrzés (fejlesztői nézet)"):
            st.write(
                f"Ez az ügyfél a valóságban "
                f"**{'felmondott' if actual else 'maradt'}**."
            )
            st.caption(
                "Éles rendszerben ez az információ nem lenne elérhető – "
                "itt csak azért látszik, mert visszatesztelt adatokon "
                "dolgozunk, és így ellenőrizhető a modell működése."
            )



def render_models_tab(method, case_id, X_test):
    """Ugyanaz az ügyfél, három modell. Ez a Rashomon-hatás demonstrációja."""
    st.subheader("Mit mond a három modell ugyanerről az ügyfélről?")
    st.caption(
        "A három modell pontossága gyakorlatilag azonos, mégis eltérő "
        "indoklást adhat. Ha egy cég a legpontosabb modellt választja, "
        "nem feltétlenül a legmegbízhatóbb magyarázatot kapja."
    )

    names = ["logreg", "tree", "xgboost"]

    with st.spinner("Mindhárom modell lekérdezése..."):
        preds = {n: get_predictions(n)[case_id] for n in names}
        expl = {n: get_explanation(n, method, case_id) for n in names}
        stabs = {n: get_stability(n, method, case_id) for n in names}

    # Előrejelzések
    cols = st.columns(3)
    for col, n in zip(cols, names):
        col.metric(MODEL_LABELS[n], f"{preds[n]:.0%}",
                   help="A modell becslése erre az ügyfélre.")

    spread = max(preds.values()) - min(preds.values())
    if spread > 0.15:
        st.warning(
            f"A három modell becslése **{spread:.0%} ponttal** tér el "
            "egymástól ennél az ügyfélnél. Ilyenkor a kockázati érték "
            "önmagában is bizonytalan."
        )
    else:
        st.success(
            f"A három modell becslése közel azonos (eltérés: {spread:.0%} pont)."
        )

    st.divider()

    # Indoklások egymás mellett
    st.markdown("**A legfontosabb tényezők modellenként**")
    table = pd.DataFrame({
        MODEL_LABELS[n]: [label(f) for f in expl[n].head(5).index]
        for n in names
    }, index=[f"{i}." for i in range(1, 6)])
    st.dataframe(table, width="stretch")

    # Egyetértés
    tops = {n: set(expl[n].head(5).index) for n in names}
    pairs = [("logreg", "tree"), ("logreg", "xgboost"), ("tree", "xgboost")]
    agree = [len(tops[a] & tops[b]) / 5 for a, b in pairs]
    mean_agree = sum(agree) / len(agree)

    top1s = {n: expl[n].abs().idxmax() for n in names}
    same_top1 = len(set(top1s.values())) == 1

    c1, c2 = st.columns(2)
    c1.metric("Modellek egyetértése", f"{mean_agree:.0%}",
              help="A top-5 tényezőkből átlagosan ennyi közös a "
                   "modellpárok között.")
    c2.metric("Ugyanaz a fő ok?", "Igen" if same_top1 else "Nem")

    if not same_top1:
        st.error(
            "**A modellek nem ugyanazt tartják a fő oknak:** "
            + " · ".join(f"{MODEL_LABELS[n]} → {label(top1s[n])}" for n in names)
            + ". Ilyenkor kerüld az egyetlen okra hivatkozást."
        )

    st.divider()

    # Stabilitás modellenként
    st.markdown("**Melyik modell indoklása a legmegbízhatóbb?**")
    st.dataframe(
        pd.DataFrame([
            {
                "Modell": MODEL_LABELS[n],
                "Kockázat": f"{preds[n]:.0%}",
                "Indoklás stabilitása": f"{stabs[n]['score']:.0%}",
                "Fő ok megmarad": f"{stabs[n]['top1_rate']:.0%}",
                "Fő ok": label(top1s[n]),
            }
            for n in names
        ]),
        hide_index=True, width="stretch",
    )
    st.caption(
        "A mérések szerint az összetettebb modellek indoklása ingatagabb. "
        "A táblázat ezt ügyfélszinten is megmutatja."
    )


@st.cache_data
def portfolio_stats(name: str) -> dict:
    _, test_df = load_frames()
    X = test_df.drop(columns=[config.TARGET])
    proba = get_predictions(name)
    at_risk = proba > 0.5
    return {
        "osszes": len(proba),
        "veszelyben": int(at_risk.sum()),
        "atlag_kockazat": float(proba.mean()),
        "bevetel_veszelyben": float(X.loc[at_risk.values, "MonthlyCharges"].sum()),
        "havi_bevetel": float(X["MonthlyCharges"].sum()),
    }


def render_portfolio_tab(model):
    """Madártávlat: mekkora a probléma, és mennyire bízhatunk a modellben."""
    st.subheader("Portfólió áttekintés")

    stats = portfolio_stats(model)
    ratio = stats["veszelyben"] / stats["osszes"]
    revenue_ratio = stats["bevetel_veszelyben"] / stats["havi_bevetel"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ügyfelek", f"{stats['osszes']:,}".replace(",", " "))
    c2.metric("Veszélyben", f"{stats['veszelyben']}",
              delta=f"{ratio:.0%} az állományból", delta_color="inverse")
    c3.metric("Átlagos kockázat", f"{stats['atlag_kockazat']:.0%}")
    c4.metric("Veszélyeztetett havi bevétel",
              f"${stats['bevetel_veszelyben']:,.0f}".replace(",", " "),
              delta=f"{revenue_ratio:.0%} a teljes bevételből",
              delta_color="inverse")

    st.caption(
        "A veszélyeztetett bevétel azoknak az ügyfeleknek a havi díja, "
        "akiknél a felmondás esélye 50% fölötti. Nem garantált veszteség: "
        "ekkora összeg forog kockán, ha nincs beavatkozás."
    )

    st.divider()

    # Modellek teljesítménye
    st.markdown("**A modellek pontossága**")
    metrics_path = config.PROJECT_ROOT / "reports" / "metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        st.dataframe(metrics, hide_index=True, width="stretch")
        auc_spread = metrics["roc_auc"].max() - metrics["roc_auc"].min()
        st.caption(
            f"A három modell találati pontossága között mindössze "
            f"**{auc_spread:.3f}** a különbség (ROC-AUC). Ez azt jelenti, "
            "hogy gyakorlatilag egyformán jók – így ha eltérő indoklást "
            "adnak, az nem abból fakad, hogy az egyik okosabb."
        )
    else:
        st.info("A metrikatáblázat nem található. Futtasd: py -m src.train")

    st.divider()

    # A kutatás eredményei
    st.markdown("**Amit a magyarázatok megbízhatóságáról tudunk**")
    st.markdown(
        """
200 ügyfélen végzett mérés alapján:

- A **SHAP** érzékeli, ha az adat pontatlan: minél nagyobb a hiba, annál
  jobban változik az indoklás. Ez a helyes viselkedés egy mérőeszköznél.
- A **LIME** válasza ugyanarra az ügyfélre futásonként is eltér, és ez az
  ingadozás akkora, hogy elnyomja az adathiba hatását. Ezért a
  megbízhatósági jelzés csak SHAP-nál értelmezhető.
- Az **összetettebb modellek indoklása ingatagabb**. A legpontosabb modell
  választása tehát a legbizonytalanabb magyarázattal jár együtt.
- A **SHAP és a LIME nem ugyanazt mondja**: a legfontosabb öt tényezőből
  átlagosan csak 3–4 közös ugyanarra az ügyfélre.
        """
    )


def main() -> None:
    model, method, case_id, proba, X_test, y_test = select_customer()

    tab1, tab2, tab3 = st.tabs([
        "Ügyfél elemzése", "Modellek összevetése", "Portfólió áttekintés",
    ])

    with tab1:
        render_customer_tab(model, method, case_id, proba, X_test, y_test)
    with tab2:
        render_models_tab(method, case_id, X_test)
    with tab3:
        render_portfolio_tab(model)


if __name__ == "__main__":
    main()
