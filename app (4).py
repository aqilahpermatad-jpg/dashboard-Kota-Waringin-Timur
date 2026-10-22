"""
Pengaruh Indikator terhadap Kemiskinan di Kab. Kotawaringin Timur
Data panel 97 kabupaten/kota, 2014-2024.
Jalankan lokal:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from linearmodels.panel import PanelOLS, PooledOLS, RandomEffects

st.set_page_config(
    page_title="Pengaruh Indikator terhadap Kemiskinan Kotim",
    page_icon="📊",
    layout="wide",
)

DATA_FILE = "dataset_fiskal.csv"
KOTA = "Kab. Kotawaringin Timur"

LABEL = {
    "KEMISKINAN": "Kemiskinan (%)",
    "KETIMPANGAN": "Ketimpangan (Gini)",
    "DBH_SDA": "DBH SDA (Dana Bagi Hasil Sumber Daya Alam)",
}
INDIKATOR = ["KETIMPANGAN", "DBH_SDA"]
SOSEKON = ["KETIMPANGAN"]
FISKAL = ["DBH_SDA"]

BIRU, MERAH, ABU, HIJAU = "#1f5fa8", "#c0392b", "#8a8f98", "#2e8b57"


def nm(c: str) -> str:
    """Label rapi; mengenali prefix ln_ untuk variabel yang di-log."""
    return "ln " + LABEL[c[3:]] if c.startswith("ln_") else LABEL.get(c, c)


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_FILE).sort_values(["Kota", "Tahun"]).reset_index(drop=True)


df = load_data()
t_min, t_max = int(df["Tahun"].min()), int(df["Tahun"].max())

# ---------------------------------------------------------------- sidebar
st.sidebar.title("Pengaturan")
st.sidebar.caption(f"Wilayah fokus: **{KOTA}**")
rentang = st.sidebar.slider("Rentang tahun", t_min, t_max, (t_min, t_max))

st.sidebar.markdown("---")
st.sidebar.subheader("Model regresi panel")
st.sidebar.markdown("Indikator: **Ketimpangan (Gini)** dan **DBH SDA**")
x_pilih = INDIKATOR
use_log = st.sidebar.checkbox("Log pada DBH SDA", value=True,
                              help="Nilai rupiah DBH SDA diubah ke logaritma natural.")
robust = st.sidebar.checkbox("SE robust (cluster per kab/kota)", value=True)

dfr = df[(df["Tahun"] >= rentang[0]) & (df["Tahun"] <= rentang[1])]
dk = dfr[dfr["Kota"] == KOTA].sort_values("Tahun")
rata = dfr.groupby("Tahun")[list(LABEL)].mean().reset_index()

st.title("Pengaruh Indikator terhadap Kemiskinan")
st.subheader(KOTA)
st.caption(
    "Seberapa besar kemiskinan di Kotawaringin Timur berkaitan dengan ketimpangan pendapatan (Gini) dan "
    "dana bagi hasil sumber daya alam (DBH SDA)? Analisis memadukan data Kotim sendiri dan "
    f"regresi panel {df['Kota'].nunique()} kab/kota ({t_min}-{t_max})."
)

# ---------------------------------------------------------------- estimasi model panel (dipakai tab 3 & 4)
if not x_pilih:
    st.warning("Pilih minimal satu indikator di sidebar.")
    st.stop()

panel = dfr.copy()
xcols = []
for v in x_pilih:
    if use_log and v in FISKAL:
        panel[f"ln_{v}"] = np.log(panel[v].clip(lower=1e-6))
        xcols.append(f"ln_{v}")
    else:
        xcols.append(v)
panel = panel.set_index(["Kota", "Tahun"])
y, X = panel["KEMISKINAN"], panel[xcols]
Xc = X.copy()
Xc.insert(0, "const", 1.0)
cov = dict(cov_type="clustered", cluster_entity=True) if robust else dict(cov_type="unadjusted")

try:
    m_pool = PooledOLS(y, Xc).fit(**cov)
    m_fe = PanelOLS(y, X, entity_effects=True).fit(**cov)
    m_re = RandomEffects(y, Xc).fit(**cov)
    fe_u = PanelOLS(y, X, entity_effects=True).fit()
    re_u = RandomEffects(y, Xc).fit()
    bd = fe_u.params - re_u.params[fe_u.params.index]
    Vd = fe_u.cov - re_u.cov.loc[fe_u.params.index, fe_u.params.index]
    haus = float(bd.values @ np.linalg.pinv(Vd.values) @ bd.values)
    haus_p = float(1 - stats.chi2.cdf(haus, len(bd)))
except Exception as e:  # noqa: BLE001
    st.error(f"Model gagal diestimasi: {e}")
    st.stop()


def tabel(m):
    t = pd.DataFrame({"Koefisien": m.params, "Std. Error": m.std_errors,
                      "t-stat": m.tstats, "p-value": m.pvalues})
    t.index = ["Konstanta" if i == "const" else nm(i) for i in t.index]
    t["Sig."] = np.select([t["p-value"] < .01, t["p-value"] < .05, t["p-value"] < .1], ["***", "**", "*"], "")
    return t


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Gambaran", "Pengaruh per Indikator (Kotim)", "Pengaruh Umum (Panel)", "Simulasi Kebijakan", "Data"]
)

# ================================================================ TAB 1: GAMBARAN
with tab1:
    last, first = dk.iloc[-1], dk.iloc[0]
    rl = rata[rata["Tahun"] == last["Tahun"]].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Kemiskinan {int(last['Tahun'])}", f"{last['KEMISKINAN']:.2f}%",
              f"{last['KEMISKINAN'] - first['KEMISKINAN']:+.2f} poin vs {int(first['Tahun'])}", delta_color="inverse")
    c2.metric("Selisih vs rata-rata kab/kota", f"{last['KEMISKINAN'] - rl['KEMISKINAN']:+.2f} poin",
              help="Negatif = lebih rendah dari rata-rata sampel pada tahun terakhir.", delta_color="off")
    c3.metric("Kemiskinan tertinggi", f"{dk['KEMISKINAN'].max():.2f}%",
              f"tahun {int(dk.loc[dk['KEMISKINAN'].idxmax(), 'Tahun'])}", delta_color="off")
    c4.metric("Kemiskinan terendah", f"{dk['KEMISKINAN'].min():.2f}%",
              f"tahun {int(dk.loc[dk['KEMISKINAN'].idxmin(), 'Tahun'])}", delta_color="off")

    st.markdown("#### Kemiskinan berdampingan dengan indikator")
    ov = st.selectbox("Bandingkan kemiskinan dengan indikator", INDIKATOR, index=0, format_func=nm)
    f = go.Figure()
    f.add_trace(go.Scatter(x=dk["Tahun"], y=dk["KEMISKINAN"], name="Kemiskinan (%)", mode="lines+markers",
                           line=dict(color=MERAH, width=3)))
    f.add_trace(go.Scatter(x=dk["Tahun"], y=dk[ov], name=nm(ov), mode="lines+markers", yaxis="y2",
                           line=dict(color=BIRU, width=2.5, dash="dot")))
    f.add_trace(go.Scatter(x=rata["Tahun"], y=rata["KEMISKINAN"], name="Kemiskinan rata-rata kab/kota",
                           mode="lines", line=dict(color=ABU, dash="dash")))
    f.update_layout(height=420, xaxis=dict(dtick=1), yaxis=dict(title="Kemiskinan (%)"),
                    yaxis2=dict(title=nm(ov), overlaying="y", side="right", showgrid=False),
                    legend=dict(orientation="h", y=1.12), margin=dict(l=10, r=10, t=10, b=10),
                    hovermode="x unified")
    st.plotly_chart(f, width="stretch")

    st.markdown("#### Kotim dibanding kab/kota lain")
    thn = st.select_slider("Tahun", options=sorted(dfr["Tahun"].unique()), value=int(dfr["Tahun"].max()))
    snap = dfr[dfr["Tahun"] == thn].copy()
    snap["Peringkat"] = snap["KEMISKINAN"].rank(ascending=False, method="min").astype(int)
    pos = snap[snap["Kota"] == KOTA].iloc[0]
    st.write(f"Tahun **{thn}**: Kotim peringkat **{int(pos['Peringkat'])} dari {len(snap)}** "
             f"(1 = kemiskinan tertinggi), kemiskinan **{pos['KEMISKINAN']:.2f}%**.")
    sx = st.selectbox("Sumbu X scatter", INDIKATOR, index=0, format_func=nm)
    snap["Wilayah"] = np.where(snap["Kota"] == KOTA, "Kotim", "Kab/kota lain")
    fs = px.scatter(snap, x=sx, y="KEMISKINAN", color="Wilayah", hover_name="Kota",
                    color_discrete_map={"Kotim": MERAH, "Kab/kota lain": "#b8c4d6"},
                    labels={sx: nm(sx), "KEMISKINAN": "Kemiskinan (%)"})
    fs.update_traces(marker=dict(size=9))
    fs.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10), legend_title_text="")
    st.plotly_chart(fs, width="stretch")

# ================================================================ TAB 2: PENGARUH PER INDIKATOR (KOTIM)
with tab2:
    st.markdown("#### Hubungan tiap indikator dengan kemiskinan, khusus Kotim")
    st.caption(f"Regresi sederhana (kemiskinan terhadap satu indikator) memakai {len(dk)} observasi tahunan Kotim. "
               "Sampel kecil, jadi baca sebagai indikasi arah dan kekuatan hubungan, bukan bukti kausal.")

    rows = []
    for v in INDIKATOR:
        if dk[v].nunique() < 2:
            continue
        r = stats.linregress(dk[v], dk["KEMISKINAN"])
        rows.append({"Indikator": nm(v), "kode": v, "Korelasi (r)": r.rvalue, "R²": r.rvalue ** 2,
                     "Kemiringan": r.slope, "p-value": r.pvalue})
    res = pd.DataFrame(rows).sort_values("Korelasi (r)")
    res["Sig."] = np.select([res["p-value"] < .01, res["p-value"] < .05, res["p-value"] < .1], ["***", "**", "*"], "")

    warna = np.where(res["p-value"] < 0.1, np.where(res["Korelasi (r)"] > 0, MERAH, HIJAU), "#b8c4d6")
    fb = go.Figure(go.Bar(x=res["Korelasi (r)"], y=res["Indikator"], orientation="h",
                          marker_color=warna, customdata=res[["p-value"]],
                          hovertemplate="%{y}<br>r = %{x:.2f}<br>p = %{customdata[0]:.3f}<extra></extra>"))
    fb.update_layout(height=60 + 26 * len(res), xaxis=dict(title="Korelasi dengan kemiskinan", range=[-1, 1]),
                     margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fb, width="stretch")
    st.caption("Merah: searah (naik bersama kemiskinan), hijau: berlawanan arah, keduanya signifikan 10%. Abu-abu: tidak signifikan.")

    st.markdown("#### Telusuri satu indikator")
    pv = st.selectbox("Indikator", INDIKATOR, format_func=nm, key="pv")
    r = stats.linregress(dk[pv], dk["KEMISKINAN"])
    a, b_, c_ = st.columns(3)
    a.metric("Korelasi (r)", f"{r.rvalue:.2f}")
    b_.metric("Kemiringan", f"{r.slope:.4f}", help="Perubahan kemiskinan (poin %) untuk kenaikan satu satuan indikator.")
    c_.metric("p-value", f"{r.pvalue:.3f}")

    xs = np.linspace(dk[pv].min(), dk[pv].max(), 50)
    fl = go.Figure()
    fl.add_trace(go.Scatter(x=dk[pv], y=dk["KEMISKINAN"], mode="markers+text", text=dk["Tahun"].astype(int),
                            textposition="top center", marker=dict(size=10, color=BIRU), name="Tahun",
                            hovertemplate="%{text}<br>" + nm(pv) + ": %{x:.2f}<br>Kemiskinan: %{y:.2f}%<extra></extra>"))
    fl.add_trace(go.Scatter(x=xs, y=r.intercept + r.slope * xs, mode="lines", name="Garis regresi",
                            line=dict(color=MERAH, width=2)))
    fl.update_layout(height=400, xaxis_title=nm(pv), yaxis_title="Kemiskinan (%)",
                     margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fl, width="stretch")

    with st.expander("Tabel lengkap"):
        st.dataframe(res.drop(columns="kode").sort_values("p-value").set_index("Indikator")
                     .style.format({"Korelasi (r)": "{:.3f}", "R²": "{:.3f}", "Kemiringan": "{:.4f}", "p-value": "{:.4f}"}),
                     width="stretch")

# ================================================================ TAB 3: PENGARUH UMUM (PANEL)
with tab3:
    st.markdown("#### Regresi panel: pengaruh indikator terhadap kemiskinan")
    st.markdown("Model: `KEMISKINAN(i,t) = a + b·X(i,t) + efek kab/kota + e(i,t)`. "
                "Variabel X dan opsi model diatur di sidebar.")
    r1, r2, r3 = st.columns(3)
    r1.metric("R² Pooled", f"{m_pool.rsquared:.3f}")
    r2.metric("R² within (FE)", f"{m_fe.rsquared:.3f}")
    r3.metric("Hausman p-value", f"{haus_p:.4f}", help="p < 0,05 menyarankan Fixed Effect.")
    st.info(f"Uji Hausman: χ² = {haus:.2f}, p = {haus_p:.4f}. "
            + ("Fixed Effect lebih tepat." if haus_p < 0.05 else "Random Effect masih dapat diterima."))

    s1, s2, s3 = st.tabs(["Fixed Effect", "Random Effect", "Pooled OLS"])
    for sub, mod in [(s1, m_fe), (s2, m_re), (s3, m_pool)]:
        with sub:
            st.dataframe(tabel(mod).style.format({"Koefisien": "{:.4f}", "Std. Error": "{:.4f}",
                                                  "t-stat": "{:.2f}", "p-value": "{:.4f}"}), width="stretch")
            st.caption("Sig.: *** p<0,01, ** p<0,05, * p<0,1")

    ci = m_fe.conf_int()
    cf = pd.DataFrame({"b": m_fe.params, "lo": ci["lower"], "hi": ci["upper"]})
    cf.index = [nm(i) for i in cf.index]
    fc = go.Figure(go.Scatter(x=cf["b"], y=cf.index, mode="markers", marker=dict(color=BIRU, size=9),
                              error_x=dict(type="data", symmetric=False, array=cf["hi"] - cf["b"],
                                           arrayminus=cf["b"] - cf["lo"])))
    fc.add_vline(x=0, line_dash="dash", line_color=ABU)
    fc.update_layout(height=80 + 45 * len(cf), xaxis_title="Pengaruh terhadap kemiskinan (poin %), 95% CI",
                     margin=dict(l=10, r=10, t=10, b=10))
    st.markdown("**Koefisien Fixed Effect**")
    st.plotly_chart(fc, width="stretch")

    st.markdown("#### Kesimpulan otomatis")
    sig = tabel(m_fe)
    sig = sig[sig["p-value"] < 0.1]
    if sig.empty:
        st.write("Tidak ada indikator yang signifikan pada tingkat 10% di model Fixed Effect ini. "
                 "Artinya, dengan spesifikasi ini, perubahan indikator tidak menjelaskan perubahan kemiskinan secara meyakinkan.")
    for n, row in sig.iterrows():
        arah = "naik" if row["Koefisien"] > 0 else "turun"
        st.write(f"- **{n}** ({row['Sig.']}): kenaikan satu satuan diasosiasikan kemiskinan {arah} "
                 f"sekitar {abs(row['Koefisien']):.2f} poin, variabel lain tetap.")
    st.caption("Ini asosiasi rata-rata seluruh sampel dan belum menangani endogenitas, jadi bukan klaim kausal.")

# ================================================================ TAB 4: SIMULASI
with tab4:
    st.markdown("#### Simulasi: jika indikator Kotim berubah, bagaimana kemiskinan?")
    st.caption("Memakai koefisien Fixed Effect dari tab sebelumnya dan titik awal data Kotim tahun terakhir. "
               "Hasilnya perkiraan berdasarkan model, bukan ramalan.")

    Xk = X.xs(KOTA, level="Kota")
    yk = y.xs(KOTA, level="Kota")
    b = m_fe.params
    alpha = float((yk - Xk[b.index] @ b).mean())
    x0 = Xk.iloc[-1]
    fit0 = alpha + float(x0[b.index] @ b)
    actual = float(yk.iloc[-1])

    delta, contrib = {}, {}
    cols = st.columns(2)
    for i, c in enumerate(b.index):
        with cols[i % 2]:
            if c.startswith("ln_"):
                p = st.slider(f"{LABEL[c[3:]]}: perubahan (%)",
                              -50, 50, 0, 5, key=f"s_{c}")
                delta[c] = float(np.log1p(p / 100))
            else:
                sd = float(df[c].std())
                step = max(round(sd / 10, 2), 0.01)
                d = st.slider(f"{nm(c)}: perubahan (satuan)", float(-2 * sd), float(2 * sd), 0.0, float(step),
                              key=f"s_{c}")
                delta[c] = d
            contrib[c] = float(b[c] * delta[c])
    skenario = fit0 + sum(contrib.values())

    k1, k2, k3 = st.columns(3)
    k1.metric(f"Kemiskinan aktual {int(dk['Tahun'].max())}", f"{actual:.2f}%")
    k2.metric("Estimasi model (tanpa perubahan)", f"{fit0:.2f}%")
    k3.metric("Estimasi skenario", f"{skenario:.2f}%", f"{skenario - fit0:+.2f} poin", delta_color="inverse")

    cdf = pd.Series(contrib).rename(index=nm).sort_values()
    fw = go.Figure(go.Bar(x=cdf.values, y=cdf.index, orientation="h",
                          marker_color=np.where(cdf.values > 0, MERAH, HIJAU)))
    fw.update_layout(height=80 + 40 * len(cdf), xaxis_title="Kontribusi ke perubahan kemiskinan (poin %)",
                     margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fw, width="stretch")
    st.caption("Merah: menaikkan kemiskinan, hijau: menurunkan. Geser slider untuk melihat indikator mana yang paling berpengaruh.")

# ================================================================ TAB 5: DATA
with tab5:
    st.dataframe(dk.set_index("Tahun")[list(LABEL)].rename(columns=LABEL), width="stretch")
    st.download_button("Unduh data Kotim (CSV)", dk.to_csv(index=False).encode("utf-8"),
                       file_name="Kotawaringin_Timur.csv", mime="text/csv")
    with st.expander("Statistik deskriptif seluruh sampel"):
        st.dataframe(dfr[list(LABEL)].rename(columns=LABEL).describe().T, width="stretch")
