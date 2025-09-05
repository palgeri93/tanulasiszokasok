# app.py — Tanulási szokások kérdőív
from __future__ import annotations

import io
import re
import unicodedata
from pathlib import Path

import pandas as pd
import streamlit as st
import altair as alt

# -------------------- Oldalbeállítás --------------------
st.set_page_config(
    page_title="Tanulási szokások kérdőív",
    page_icon="📚",
    layout="centered",
)

BASE = Path(__file__).parent.resolve()

# -------------------- Segédfüggvények --------------------
def _ascii_fold(s: str) -> str:
    """Ékezetek eltávolítása biztonságosan (NFD + combining mark szűrés)."""
    if not isinstance(s, str):
        s = str(s)
    return "".join(
        ch for ch in unicodedata.normalize("NFD", s)
        if unicodedata.category(ch) != "Mn"
    )

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    def norm(s: str) -> str:
        s2 = _ascii_fold(str(s)).lower().strip()
        s2 = re.sub(r"\s+", "_", s2)
        return s2
    return df.rename(columns={c: norm(c) for c in df.columns})

def _find_xlsx_for_learning() -> Path | None:
    """Megpróbáljuk megtalálni a tanulási szokások .xlsx fájlt rugalmasan."""
    candidates = list(BASE.glob("*.xlsx"))
    exact = BASE / "tanulasi szokasok.xlsx"
    if exact.exists():
        return exact
    def tokens(p: Path) -> set[str]:
        t = _ascii_fold(p.stem).lower()
        t = re.sub(r"[^0-9a-z]+", " ", t)
        return set(t.split())
    want = {"tanulasi", "szokasok"}
    scored = [p for p in candidates if want.issubset(tokens(p))]
    return scored[0] if scored else (candidates[0] if candidates else None)

# -------------------- Betöltés --------------------
@st.cache_data(show_spinner=True)
def load_bank(path: Path) -> pd.DataFrame:
    """Beolvassa a kérdőívet. Elvárt oszlopok: kategória (A oszlop), kérdés, inverz."""
    raw = pd.read_excel(path)
    df = _norm_cols(raw)

    kerdes_col = None
    for c in ["kerdes", "kerdes_szoveg", "allitas", "item", "szoveg"]:
        if c in df.columns:
            kerdes_col = c
            break
    kat_col = "kategoria" if "kategoria" in df.columns else None
    inv_col = "inverz" if "inverz" in df.columns else None

    if kerdes_col is None or kat_col is None or inv_col is None:
        raise ValueError("Az Excelben a 'kategória', 'kérdés' és 'inverz' oszlopok szükségesek.")

    inv_bool = (
        df[inv_col].astype(str).str.strip().str.lower()
        .isin(["igen", "true", "1", "y", "yes"])
    )

    out = pd.DataFrame({
        "kerdes": df[kerdes_col].astype(str).str.strip(),
        "kategoria": df[kat_col].astype(str).str.strip(),
        "inverse": inv_bool.fillna(False),
    })
    return out[out["kerdes"].str.len() > 0].reset_index(drop=True)

# -------------------- UI – fejléc, űrlap --------------------
st.markdown("## Tanulási szokások kérdőív")
st.write("Add meg a neved és az osztályod, majd töltsd ki az állításokat az alábbi skálán.")

with st.form("meta_form", clear_on_submit=False):
    nev = st.text_input("Név", value=st.session_state.get("nev", "")).strip()
    osztaly = st.text_input("Osztály", value=st.session_state.get("osztaly", "")).strip()
    if st.form_submit_button("Mentés"):
        st.session_state["nev"] = nev
        st.session_state["osztaly"] = osztaly

xlsx_path = _find_xlsx_for_learning()
if not xlsx_path or not xlsx_path.exists():
    st.error("Nem találom a **tanulasi szokasok.xlsx** fájlt a projekt gyökerében.")
    st.stop()

try:
    bank_df = load_bank(xlsx_path)
except Exception as e:
    st.error(f"Hiba a kérdésbank betöltésénél: {e}")
    st.stop()

# -------------------- Kitöltő felület --------------------
if not st.session_state.get("nev") or not st.session_state.get("osztaly"):
    st.warning("A folytatáshoz add meg a **Név** és **Osztály** mezőket a fenti űrlapon.")
    st.stop()

# Nagy betűs, bevezető szöveg a kitöltés elé
st.markdown(
    """
    <div style="font-size:1.05rem; font-weight:600; line-height:1.5; margin: 0.5rem 0 0.75rem;">
    Olvasd el figyelmesen az alábbi mondatokat. Döntsd el, hogy az öt válasz közül melyik jellemző rád, és azt jelöld meg!<br>
    A hármas választ lehetőleg ritkán használd, csak akkor, ha semmiképpen sem tudsz dönteni. Jó munkát kívánok!
    </div>
    """,
    unsafe_allow_html=True,
)

LIKERT_OPCIOK = [
    "Egyáltalán nem jellemző",
    "Inkább nem jellemző",
    "Részben jellemző",
    "Inkább jellemző",
    "Teljesen jellemző",
]

if "valaszok" not in st.session_state:
    st.session_state["valaszok"] = {}
valaszok: dict[int, int] = st.session_state["valaszok"]

st.divider()
st.write("Jelöld meg, mennyire jellemzőek rád az alábbi állítások.")

for i, sor in bank_df.iterrows():
    kerdes = sor["kerdes"]
    st.markdown(
        f'<div style="font-weight:700; font-size:1.15rem; margin-top:0.5rem;">{i+1}. {kerdes}</div>',
        unsafe_allow_html=True,
    )
    key = f"q_{i}"
    default_idx = valaszok.get(i, None)
    idx = st.radio(
        label="",
        options=list(range(len(LIKERT_OPCIOK))),
        format_func=lambda k: LIKERT_OPCIOK[k],
        index=default_idx if default_idx is not None else None,
        horizontal=False,
        key=key,
        label_visibility="collapsed",
    )
    if idx is not None:
        valaszok[i] = idx

osszes_kerdes = len(bank_df)
megvalaszolt = len(valaszok)
if megvalaszolt < osszes_kerdes:
    st.warning(f"Még **{osszes_kerdes - megvalaszolt}** kérdésre nem válaszoltál.")
    st.stop()

# -------------------- Pontszámítás --------------------
bank_df["raw"] = [valaszok[i] + 1 for i in range(osszes_kerdes)]  # 1..5
bank_df["score"] = bank_df.apply(lambda r: 6 - r["raw"] if r["inverse"] else r["raw"], axis=1)

# -------------------- Kategória-összesítés (átlag) --------------------
kat_agg = (
    bank_df.groupby("kategoria")["score"]
    .agg(["count", "sum", "mean"])
    .reset_index()
    .rename(columns={"count": "tételszám", "sum": "összpont", "mean": "átlag"})
    .sort_values("kategoria")
)

st.divider()
st.markdown("### Eredmények (kategóriaátlagok)")

# Táblázat (csak tételszám és átlag)
st.dataframe(
    kat_agg[["kategoria", "tételszám", "átlag"]].rename(columns={"kategoria": "Kategória"}),
    hide_index=True,
    use_container_width=True,
)
# --- FÜGGŐLEGES DIAGRAM: x = kategóriák, y = átlag; felirat az oszlop felett ---
tick_values = [x / 2 for x in range(2, 11)]  # 1.0, 1.5, ..., 5.0

base = alt.Chart(kat_agg).encode(
    x=alt.X(
        "kategoria:N",
        title="Kategória",
        axis=alt.Axis(
            orient="bottom",
            labelFontWeight="bold",   # félkövér kategóriafelirat
            labelAngle=0,             # vízszintes
            labelPadding=10,          # térköz a tengely és felirat közt
            labelLimit=1000,          # ne vágja le
            labelOverlap=False,       # NE rejtse el ütközésnél sem
            ticks=True
        ),
    ),
    y=alt.Y(
        "átlag:Q",
        title="Átlagpont",
        scale=alt.Scale(domain=[1, 5]),
        axis=alt.Axis(values=tick_values)
    ),
    tooltip=["kategoria", alt.Tooltip("átlag:Q", format=".2f"), "tételszám"],
)

bars = base.mark_bar()
labels = base.mark_text(dy=-6).encode(text=alt.Text("átlag:Q", format=".2f"))

chart = (bars + labels).properties(
    height=420,
    padding={"left": 5, "right": 5, "top": 10, "bottom": 110},  # extra hely az x feliratoknak
).configure_axisX(
    labelFontWeight="bold"  # redundáns biztosítás
)

st.altair_chart(chart, use_container_width=True)
# -------------------- Letöltés XLSX --------------------
st.divider()
st.subheader("Riport letöltése")

valaszok_long = bank_df[["kategoria", "kerdes", "inverse", "raw", "score"]].copy()
valaszok_long.rename(columns={
    "kategoria": "Kategória",
    "kerdes": "Kérdés",
    "inverse": "Inverz kérdés?",
    "raw": "Jelölt érték (1..5)",
    "score": "Pont (inverz után)",
}, inplace=True)

wide = bank_df.groupby("kategoria")["score"].mean().round(2).to_frame().T
wide.index = [0]

buf = io.BytesIO()
with pd.ExcelWriter(buf, engine="xlsxwriter") as wr:
    valaszok_long.to_excel(wr, sheet_name="valaszok", index=False)
    kat_agg.to_excel(wr, sheet_name="kategoriak", index=False)
    wide.to_excel(wr, sheet_name="kategoriak_wide_atlag", index=False)

fnev = f"tanulasi_szokasok_eredmeny_{(st.session_state.get('nev') or 'tanulo').replace(' ', '_')}.xlsx"
st.download_button(
    "Eredmény letöltése (XLSX)",
    data=buf.getvalue(),
    file_name=fnev,
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

# -------------------- Lábjegyzet --------------------
st.markdown(
    '<div style="text-align:center; color:#666; margin-top:1.5rem;">készítette Pálfi Gergő 2025.</div>',
    unsafe_allow_html=True,
)
