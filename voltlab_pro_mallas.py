# -*- coding: utf-8 -*-
"""
VoltLab Pro — Simulador AC + Subida de Imagen + Método de Mallas

Instalación:
    pip install streamlit numpy matplotlib pillow

Ejecución:
    streamlit run voltlab_pro_mallas.py
"""

import math
import cmath
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image, UnidentifiedImageError
from numpy.linalg import LinAlgError


# ==================================================
# CONFIGURACIÓN DE STREAMLIT
# Debe ejecutarse antes de otros comandos st.*
# ==================================================
st.set_page_config(
    page_title="VoltLab Pro — Mallas",
    page_icon="⚡",
    layout="wide",
)


# ==================================================
# UTILIDADES
# ==================================================
def polar(z: complex) -> str:
    """Representa un fasor en forma polar."""
    z = complex(z)
    return f"{abs(z):.4f} ∠ {math.degrees(cmath.phase(z)):.3f}°"


def rectangular(z: complex) -> str:
    """Representa un número complejo en forma rectangular."""
    z = complex(z)
    sign = "+" if z.imag >= 0 else "−"
    return f"{z.real:.4f} {sign} j{abs(z.imag):.4f}"


def format_eng(value: float, unit: str = "", sig: int = 4) -> str:
    """Formato de ingeniería con prefijos SI."""
    value = float(value)

    if not math.isfinite(value):
        return f"{value} {unit}".strip()

    if abs(value) < 1e-30:
        return f"0 {unit}".strip()

    prefixes = {
        -12: "p", -9: "n", -6: "µ", -3: "m",
        0: "", 3: "k", 6: "M", 9: "G", 12: "T"
    }

    exponent = int(math.floor(math.log10(abs(value)) / 3) * 3)
    exponent = max(-12, min(12, exponent))
    return f"{value / (10 ** exponent):.{sig}g} {prefixes[exponent]}{unit}".strip()


def impedance(R: float, L: float, C: float, omega: float) -> complex:
    """Calcula Z = R + j(wL - 1/(wC))."""
    if R < 0 or L < 0 or C < 0:
        raise ValueError("R, L y C no pueden ser negativos.")

    x_l = omega * L
    x_c = 1.0 / (omega * C) if C > 0 else 0.0
    return complex(R, x_l - x_c)


# ==================================================
# SUBIDA DE IMAGEN
# ==================================================
def image_uploader():
    st.markdown("### 📸 Sube la foto de tu circuito")

    uploaded_file = st.file_uploader(
        "Sube una imagen (JPG, JPEG o PNG)",
        type=["jpg", "jpeg", "png"],
        help="Procura que el esquema se vea completo, enfocado y con buena iluminación.",
        key="circuit_image",
    )

    if uploaded_file is None:
        return None, None

    try:
        image = Image.open(uploaded_file)
        image.load()
        st.image(
            image,
            caption="✅ Imagen cargada correctamente",
            use_container_width=True,
        )
        return uploaded_file, image
    except (UnidentifiedImageError, OSError, ValueError):
        st.error("No se pudo abrir la imagen. Intenta subir otro archivo JPG o PNG.")
        return None, None


# ==================================================
# CONFIGURACIÓN DE MALLAS
# ==================================================
def mesh_config_wizard() -> dict[str, Any]:
    st.markdown("---")
    st.markdown("### ⚙️ Configuración del circuito por mallas")

    n = int(st.number_input(
        "Número de mallas del circuito",
        min_value=1,
        max_value=5,
        value=2,
        step=1,
        key="number_meshes",
    ))

    frequency = float(st.number_input(
        "Frecuencia del circuito (Hz)",
        min_value=0.001,
        value=60.0,
        step=1.0,
        format="%.3f",
        key="frequency",
    ))

    omega = 2.0 * math.pi * frequency
    st.info(f"ω = 2πf = 2π({frequency:g}) = **{omega:.4f} rad/s**")

    st.markdown("#### 📐 Definir matriz de impedancias [Z]")
    st.caption(
        "Zii es la impedancia total propia de la malla i. "
        "Zij es el negativo de la impedancia compartida entre las mallas i y j."
    )

    Z = [[0j for _ in range(n)] for _ in range(n)]

    for i in range(n):
        st.markdown(f"**Malla {i + 1}: impedancia propia**")
        cols = st.columns(3)

        with cols[0]:
            Rii = st.number_input(
                f"R total malla {i + 1} (Ω)",
                min_value=0.0,
                value=10.0 if i == 0 else 10.0,
                step=1.0,
                key=f"R_mesh_{i}",
            )
        with cols[1]:
            Lii = st.number_input(
                f"L total malla {i + 1} (H)",
                min_value=0.0,
                value=0.01,
                step=0.001,
                format="%.4f",
                key=f"L_mesh_{i}",
            )
        with cols[2]:
            Cii = st.number_input(
                f"C total malla {i + 1} (F)",
                min_value=0.0,
                value=0.0,
                step=0.0001,
                format="%.5f",
                key=f"C_mesh_{i}",
            )

        Zii = impedance(float(Rii), float(Lii), float(Cii), omega)
        Z[i][i] = Zii
        st.caption(f"Z{i + 1}{i + 1} = {rectangular(Zii)} Ω")

        for j in range(i + 1, n):
            st.markdown(f"🔗 **Impedancia compartida entre mallas {i + 1} y {j + 1}**")
            common_cols = st.columns(3)

            with common_cols[0]:
                Rij = st.number_input(
                    f"R común {i + 1}-{j + 1} (Ω)",
                    min_value=0.0,
                    value=0.0,
                    step=1.0,
                    key=f"R_common_{i}_{j}",
                )
            with common_cols[1]:
                Lij = st.number_input(
                    f"L común {i + 1}-{j + 1} (H)",
                    min_value=0.0,
                    value=0.0,
                    step=0.001,
                    format="%.4f",
                    key=f"L_common_{i}_{j}",
                )
            with common_cols[2]:
                Cij = st.number_input(
                    f"C común {i + 1}-{j + 1} (F)",
                    min_value=0.0,
                    value=0.0,
                    step=0.0001,
                    format="%.5f",
                    key=f"C_common_{i}_{j}",
                )

            Zij = impedance(float(Rij), float(Lij), float(Cij), omega)
            Z[i][j] = -Zij
            Z[j][i] = -Zij
            st.caption(
                f"Z{i + 1}{j + 1} = Z{j + 1}{i + 1} = "
                f"−({rectangular(Zij)}) Ω"
            )

    st.markdown("#### 🔌 Fuentes de voltaje por malla [V RMS]")
    st.caption(
        "Ingresa el valor eficaz (RMS) y el ángulo de fase de cada fuente. "
        "La convención de signos debe coincidir con el sentido de las corrientes de malla."
    )

    V = []
    for i in range(n):
        voltage_cols = st.columns(2)

        with voltage_cols[0]:
            voltage_rms = st.number_input(
                f"Voltaje RMS malla {i + 1} (V)",
                value=10.0,
                step=1.0,
                key=f"voltage_{i}",
            )
        with voltage_cols[1]:
            phase_deg = st.number_input(
                f"Fase malla {i + 1} (°)",
                value=0.0,
                step=5.0,
                key=f"phase_{i}",
            )

        Vi = float(voltage_rms) * cmath.exp(1j * math.radians(float(phase_deg)))
        V.append(Vi)
        st.caption(f"V{i + 1} = {polar(Vi)} V RMS")

    return {
        "n_mallas": n,
        "f": frequency,
        "w": omega,
        "Z_matrix": Z,
        "V_vector": V,
    }


# ==================================================
# RESOLUCIÓN POR MÉTODO DE MALLAS
# ==================================================
def solve_by_meshes(mesh_config: dict[str, Any]) -> dict[str, Any]:
    n = mesh_config["n_mallas"]
    Z = np.asarray(mesh_config["Z_matrix"], dtype=complex)
    V = np.asarray(mesh_config["V_vector"], dtype=complex)
    frequency = float(mesh_config["f"])
    omega = float(mesh_config["w"])

    if Z.shape != (n, n) or V.shape != (n,):
        raise ValueError("Las dimensiones de la matriz Z o del vector V no son correctas.")

    if not np.all(np.isfinite(Z)) or not np.all(np.isfinite(V)):
        raise ValueError("Hay valores no válidos en las impedancias o en las fuentes.")

    lines = [
        "=" * 68,
        "RESOLUCIÓN POR EL MÉTODO DE CORRIENTES DE MALLA",
        "=" * 68,
        f"Número de mallas: {n}",
        f"Frecuencia: f = {format_eng(frequency, 'Hz')}",
        f"Velocidad angular: ω = {omega:.4f} rad/s",
        "",
        "PASO 1 — ECUACIONES DE MALLA",
        "Sistema matricial: [Z][I] = [V]",
        "Para cada malla: Z_i1 I_1 + Z_i2 I_2 + ... + Z_in I_n = V_i",
        "",
    ]

    for i in range(n):
        terms = []
        for j in range(n):
            z_value = Z[i, j]
            if abs(z_value) > 1e-12:
                terms.append(f"({rectangular(z_value)})·I{j + 1}")

        equation = " + ".join(terms) if terms else "0"
        lines.append(f"Ecuación {i + 1}: {equation} = {polar(V[i])} V")

    lines.extend(["", "PASO 2 — MATRIZ DE IMPEDANCIAS [Z] (Ω)"])
    for i in range(n):
        row = " | ".join(rectangular(Z[i, j]) for j in range(n))
        lines.append(f"Fila {i + 1}: {row}")

    lines.extend(["", "PASO 3 — VECTOR DE FUENTES [V]"])
    for i in range(n):
        lines.append(f"V{i + 1} = {polar(V[i])} V RMS")

    try:
        determinant = np.linalg.det(Z)
        condition_number = np.linalg.cond(Z)

        lines.extend([
            "",
            "PASO 4 — RESOLUCIÓN DEL SISTEMA",
            f"Determinante det(Z) = {rectangular(complex(determinant))}",
            f"Número de condición = {condition_number:.4g}",
        ])

        if not np.isfinite(condition_number) or condition_number > 1e12:
            raise ValueError(
                "La matriz de impedancias es singular o está mal condicionada. "
                "Revisa las impedancias y las conexiones."
            )

        mesh_currents = np.linalg.solve(Z, V)

        lines.extend(["", "CORRIENTES DE MALLA (valores RMS)"])
        for i, current in enumerate(mesh_currents):
            lines.append(
                f"I{i + 1} = {polar(current)} A = {rectangular(current)} A"
            )

        # Potencia compleja total absorbida por la red:
        # S = suma(V_i * conjugado(I_i)) = P + jQ.
        # Los fasores de entrada se interpretan como valores RMS.
        complex_power = np.sum(V * np.conjugate(mesh_currents))
        total_P = float(complex_power.real)
        total_Q = float(complex_power.imag)
        total_S = float(abs(complex_power))
        power_factor = total_P / total_S if total_S > 1e-12 else 0.0

        lines.extend([
            "",
            "PASO 5 — POTENCIA COMPLEJA TOTAL",
            "S = P + jQ = Σ(V_i · conjugado(I_i))",
            f"S = {rectangular(complex_power)} VA",
            f"Potencia activa P = {total_P:.4f} W",
            f"Potencia reactiva Q = {total_Q:.4f} var",
            f"Potencia aparente |S| = {total_S:.4f} VA",
            f"Factor de potencia |FP| = {abs(power_factor):.4f}",
        ])

        if total_Q > 1e-9:
            circuit_type = "INDUCTIVO (Q positiva)"
        elif total_Q < -1e-9:
            circuit_type = "CAPACITIVO (Q negativa)"
        else:
            circuit_type = "RESISTIVO (Q aproximadamente cero)"

        lines.extend([
            f"Tipo de comportamiento: {circuit_type}",
            "",
            "NOTA: Los resultados de potencia se calculan usando fasores RMS.",
            "La potencia total corresponde al sistema definido por la matriz [Z].",
            "=" * 68,
        ])

        return {
            "n": n,
            "f": frequency,
            "Z": Z,
            "V": V,
            "I_mallas": mesh_currents,
            "P_total": total_P,
            "Q_total": total_Q,
            "S_total": total_S,
            "S_complex": complex_power,
            "fp": abs(power_factor),
            "circuit_type": circuit_type,
            "steps": "\n".join(lines),
        }

    except (LinAlgError, ValueError, FloatingPointError) as exc:
        raise ValueError(
            f"No se pudo resolver el circuito: {exc}\n\n"
            "Comprueba que cada malla tenga una impedancia definida, que los "
            "valores sean correctos y que la matriz no sea singular."
        ) from exc


# ==================================================
# GRÁFICAS
# ==================================================
def plot_mesh_results(mesh_result: dict[str, Any]):
    currents = mesh_result["I_mallas"]
    n = len(currents)
    frequency = mesh_result["f"]
    omega = 2.0 * math.pi * frequency

    time = np.linspace(0.0, 3.0 / frequency, 1000)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=110)

    colors = ["#2563eb", "#dc2626", "#f59e0b", "#10b981", "#7c3aed"]

    for i, current in enumerate(currents):
        # Si el fasor está expresado en RMS, el valor pico es sqrt(2) * RMS.
        current_time = (
            math.sqrt(2.0)
            * abs(current)
            * np.cos(omega * time + cmath.phase(current))
        )

        ax.plot(
            time * 1000.0,
            current_time,
            label=f"I{i + 1}: {polar(current)} A RMS",
            color=colors[i % len(colors)],
            linewidth=2.0,
        )

    ax.set_title("Corrientes de malla — Dominio del tiempo")
    ax.set_xlabel("Tiempo (ms)")
    ax.set_ylabel("Corriente instantánea (A)")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.axhline(0, color="#475569", linewidth=0.8)
    ax.legend(loc="upper right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    return fig


# ==================================================
# INTERFAZ PRINCIPAL
# ==================================================
st.title("⚡ VOLTLAB PRO")
st.subheader("Simulador de circuitos AC — Método de corrientes de malla")
st.markdown("---")

if "mesh_result" not in st.session_state:
    st.session_state.mesh_result = None

tab_upload, tab_config, tab_results = st.tabs([
    "📸 1. Subir ejercicio",
    "⚙️ 2. Configurar mallas",
    "📐 3. Ver resultados",
])


# ---------- Pestaña 1 ----------
with tab_upload:
    image_uploader()
    st.info(
        "💡 Sube una fotografía clara como referencia. "
        "Después ingresa manualmente las impedancias y fuentes en la pestaña 2. "
        "La imagen todavía no se analiza automáticamente."
    )


# ---------- Pestaña 2 ----------
with tab_config:
    if st.button("🔄 Reiniciar resultados", key="reset_results"):
        st.session_state.mesh_result = None
        st.rerun()

    config = mesh_config_wizard()

    st.markdown("---")
    if st.button(
        "✅ Resolver circuito por MALLAS",
        type="primary",
        use_container_width=True,
        key="solve_meshes",
    ):
        try:
            with st.spinner("Resolviendo el sistema de ecuaciones..."):
                result = solve_by_meshes(config)
                st.session_state.mesh_result = result

            st.success("¡Circuito resuelto correctamente! Abre la pestaña 3 para ver los resultados.")

        except Exception as exc:
            st.error(str(exc))


# ---------- Pestaña 3 ----------
with tab_results:
    result = st.session_state.mesh_result

    if result is not None:
        st.success("Resultados disponibles")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Potencia activa", f"{result['P_total']:.4f} W")
        col2.metric("Potencia reactiva", f"{result['Q_total']:.4f} var")
        col3.metric("Potencia aparente", f"{result['S_total']:.4f} VA")
        col4.metric("Factor de potencia", f"{result['fp']:.4f}")

        st.write(f"**Comportamiento del circuito:** {result['circuit_type']}")

        st.markdown("### Desarrollo paso a paso")
        st.code(result["steps"], language="text")

        st.download_button(
            "💾 Descargar resolución completa (.txt)",
            data=result["steps"],
            file_name="resolucion_mallas.txt",
            mime="text/plain",
            key="download_results",
        )

        st.markdown("### Gráfica de corrientes")
        figure = plot_mesh_results(result)
        st.pyplot(figure, use_container_width=True)
        plt.close(figure)

    else:
        st.info(
            "👈 Ve a la pestaña 2, configura tu circuito y pulsa "
            "**Resolver circuito por MALLAS**."
        )
