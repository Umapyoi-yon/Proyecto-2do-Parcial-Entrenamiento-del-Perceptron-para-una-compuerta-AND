"""app.py
Capa de presentación (Streamlit) del Perceptrón Monocapa – Compuerta AND.
Ejecutar con:  streamlit run app.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from pdf_generator import generate_pdf
from perceptron import AND_DATASET, Perceptron

st.set_page_config(page_title="Perceptrón Monocapa – AND", page_icon="🧠", layout="wide")

"""Formato completo sin truncar (hasta 10 decimales, sin ceros sobrantes)."""
# ------------------------------------------------------------------- utilidades
def fmt(v: float) -> str:
    
    s = f"{v:.10f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s

"""REQ 1: esquema dinámico del perceptrón (los valores cambian con los parámetros)."""
def network_svg(w1: float, w2: float, b: float,
                x1=None, x2=None, z=None, y=None) -> str:
    
    def txt(v):
        return "?" if v is None else fmt(v)
    out_color = "#9aa5b1" if y is None else ("#2e9e5b" if y == 1 else "#d1495b")
    return f"""
<svg viewBox="0 0 760 340" xmlns="http://www.w3.org/2000/svg" font-family="Arial, sans-serif" style="width:100%;max-width:900px">
  <defs><marker id="a" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">
    <path d="M0,0 L10,4 L0,8 z" fill="#445"/></marker></defs>
  <!-- entradas -->
  <circle cx="70" cy="90" r="30" fill="#e8f0fe" stroke="#1f3a5f" stroke-width="2"/>
  <text x="70" y="86" text-anchor="middle" font-size="16" font-weight="bold">x₁</text>
  <text x="70" y="104" text-anchor="middle" font-size="13">{txt(x1)}</text>
  <circle cx="70" cy="230" r="30" fill="#e8f0fe" stroke="#1f3a5f" stroke-width="2"/>
  <text x="70" y="226" text-anchor="middle" font-size="16" font-weight="bold">x₂</text>
  <text x="70" y="244" text-anchor="middle" font-size="13">{txt(x2)}</text>
  <!-- sesgo -->
  <circle cx="300" cy="300" r="22" fill="#fff4e0" stroke="#c77700" stroke-width="2"/>
  <text x="300" y="305" text-anchor="middle" font-size="15" font-weight="bold">b</text>
  <text x="300" y="332" text-anchor="middle" font-size="13" fill="#c77700">{fmt(b)}</text>
  <!-- conexiones -->
  <line x1="100" y1="95" x2="388" y2="150" stroke="#445" stroke-width="2" marker-end="url(#a)"/>
  <line x1="100" y1="225" x2="388" y2="185" stroke="#445" stroke-width="2" marker-end="url(#a)"/>
  <line x1="318" y1="284" x2="396" y2="196" stroke="#c77700" stroke-width="2" marker-end="url(#a)"/>
  <rect x="150" y="82" width="120" height="22" rx="4" fill="#fff" stroke="#ccd"/>
  <text x="210" y="98" text-anchor="middle" font-size="13">w₁ = {fmt(w1)}</text>
  <rect x="150" y="205" width="120" height="22" rx="4" fill="#fff" stroke="#ccd"/>
  <text x="210" y="221" text-anchor="middle" font-size="13">w₂ = {fmt(w2)}</text>
  <!-- suma -->
  <circle cx="420" cy="170" r="36" fill="#f3f3f3" stroke="#1f3a5f" stroke-width="2"/>
  <text x="420" y="178" text-anchor="middle" font-size="28">Σ</text>
  <text x="420" y="225" text-anchor="middle" font-size="13">z = {txt(z)}</text>
  <line x1="456" y1="170" x2="514" y2="170" stroke="#445" stroke-width="2" marker-end="url(#a)"/>
  <!-- escalón -->
  <rect x="520" y="130" width="90" height="80" rx="8" fill="#f3f3f3" stroke="#1f3a5f" stroke-width="2"/>
  <polyline points="535,190 565,190 565,150 595,150" fill="none" stroke="#1f3a5f" stroke-width="3"/>
  <text x="565" y="228" text-anchor="middle" font-size="12">Escalón f(z)</text>
  <line x1="610" y1="170" x2="662" y2="170" stroke="#445" stroke-width="2" marker-end="url(#a)"/>
  <!-- salida -->
  <circle cx="696" cy="170" r="32" fill="{out_color}" stroke="#222" stroke-width="2"/>
  <text x="696" y="165" text-anchor="middle" font-size="15" font-weight="bold" fill="#fff">y</text>
  <text x="696" y="187" text-anchor="middle" font-size="18" fill="#fff">{'?' if y is None else y}</text>
</svg>"""


def show_diagram(w1, w2, b, x1=None, x2=None, z=None, y=None):
    st.image(network_svg(w1, w2, b, x1, x2, z, y).strip(), width=900)


# ---------------------------------------------------------------------- sidebar
st.sidebar.header("⚙️ Hiperparámetros iniciales")
w1_0 = st.sidebar.number_input("Peso inicial w₁", value=0.0, step=0.1, format="%.4f")
w2_0 = st.sidebar.number_input("Peso inicial w₂", value=0.0, step=0.1, format="%.4f")
b_0 = st.sidebar.number_input("Bias inicial b", value=0.0, step=0.1, format="%.4f")
eta = st.sidebar.slider("Tasa de aprendizaje η", 0.01, 1.0, 0.1, 0.01)
max_epochs = st.sidebar.number_input("Límite máximo de épocas", min_value=1, max_value=500, value=20, step=1)
train_clicked = st.sidebar.button("🚀 Entrenar perceptrón", type="primary", width="stretch")

if train_clicked:
    st.session_state["model"] = Perceptron(w1_0, w2_0, b_0, eta, int(max_epochs)).fit(AND_DATASET)

model: Perceptron | None = st.session_state.get("model")

# ----------------------------------------------------------------------- cabecera
st.title("🧠 Perceptrón Monocapa – Compuerta AND")
st.caption("Simulación pedagógica del algoritmo de aprendizaje con la regla delta.")

# REQ 1 – esquema (placeholder: se rellena al final para reflejar la inferencia)
st.subheader("1. Esquema de la red")
diagram_slot = st.empty()

st.markdown("**Datos de entrenamiento (AND):**")
st.dataframe(pd.DataFrame(AND_DATASET, columns=["x₁", "x₂", "t (deseada)"]),
             hide_index=True, width="content")

if model is None:
    with diagram_slot.container():
        show_diagram(w1_0, w2_0, b_0)
    st.info("Configura los hiperparámetros en la barra lateral y pulsa **Entrenar perceptrón**.")
    st.stop()

# ----------------------------------------------------------------------- resultado
if model.converged:
    st.success(f"✅ Convergió en la época {len(model.epochs)} "
               f"(w₁ = {fmt(model.w1)}, w₂ = {fmt(model.w2)}, b = {fmt(model.b)}).")
else:
    st.warning(f"⚠️ Se alcanzó el límite de {model.max_epochs} épocas sin converger. "
               "Prueba con otros pesos iniciales, otra tasa de aprendizaje o más épocas.")

# REQ 4 – tabla resumen
st.subheader("2. Resumen por épocas")
df = pd.DataFrame(model.epoch_table()).rename(columns={
    "epoch": "Época", "w1": "w₁", "w2": "w₂", "b": "b", "errors": "Errores"})
st.dataframe(df, hide_index=True, width="stretch")

# REQ 5 – PDF
st.download_button("📄 Descargar informe PDF", data=generate_pdf(model),
                   file_name="perceptron_AND_informe.pdf", mime="application/pdf")

# REQ 3 – procedimiento paso a paso
st.subheader("3. Procedimiento matemático paso a paso")
st.markdown("Para cada muestra: calcular la suma ponderada, evaluar el escalón, obtener el error y "
            "actualizar con la regla delta (para el bias la entrada es 1).")
for ep in model.epochs:
    label = f"Época {ep.epoch} — errores: {ep.errors}"
    with st.expander(label, expanded=(ep.epoch == 1)):
        for s in model.steps_of_epoch(ep.epoch):
            st.markdown(f"**Muestra {s.sample}:** $x_1={fmt(s.x1)},\\ x_2={fmt(s.x2)},\\ t={s.t}$")
            st.latex(
                rf"""\begin{{aligned}}
z &= (x_1 \cdot w_1) + (x_2 \cdot w_2) + b = ({fmt(s.x1)} \cdot {fmt(s.w1_prev)}) + ({fmt(s.x2)} \cdot {fmt(s.w2_prev)}) + {fmt(s.b_prev)} = {fmt(s.z)} \\
y &= f(z) = f({fmt(s.z)}) = {s.y} \qquad e = t - y = {s.t} - {s.y} = {s.e} \\
w_1^{{(n)}} &= w_1^{{(prev)}} + \eta \cdot e \cdot x_1 = {fmt(s.w1_prev)} + {fmt(model.eta)} \cdot {s.e} \cdot {fmt(s.x1)} = {fmt(s.w1_new)} \\
w_2^{{(n)}} &= w_2^{{(prev)}} + \eta \cdot e \cdot x_2 = {fmt(s.w2_prev)} + {fmt(model.eta)} \cdot {s.e} \cdot {fmt(s.x2)} = {fmt(s.w2_new)} \\
b^{{(n)}} &= b^{{(prev)}} + \eta \cdot e \cdot 1 = {fmt(s.b_prev)} + {fmt(model.eta)} \cdot {s.e} \cdot 1 = {fmt(s.b_new)}
\end{{aligned}}""")
            st.divider()

# REQ 6 – inferencia
st.subheader("4. Módulo de inferencia")
st.markdown("Ingresa cualquier combinación $(x_1, x_2)$; se evalúa con los pesos finales y el resultado "
            "se refleja en el esquema de arriba.")
c1, c2 = st.columns(2)
tx1 = c1.number_input("x₁", value=1.0, step=0.1, format="%.2f", key="inf_x1")
tx2 = c2.number_input("x₂", value=1.0, step=0.1, format="%.2f", key="inf_x2")
z, y = model.predict(tx1, tx2)
st.latex(rf"z = ({fmt(tx1)} \cdot {fmt(model.w1)}) + ({fmt(tx2)} \cdot {fmt(model.w2)}) + {fmt(model.b)} = {fmt(z)}"
         rf"\quad\Rightarrow\quad y = f({fmt(z)}) = {y}")
expected = 1 if (tx1 == 1 and tx2 == 1) else 0
if tx1 in (0, 1) and tx2 in (0, 1):
    (st.success if y == expected else st.error)(
        f"Salida predicha: **{y}** — valor esperado de AND: **{expected}**")
else:
    st.info(f"Salida predicha: **{y}** (entrada fuera del dominio binario de la compuerta AND).")

with diagram_slot.container():
    show_diagram(model.w1, model.w2, model.b, tx1, tx2, z, y)