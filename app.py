# -*- coding: utf-8 -*-
"""
Detector de objetos con YOLO + Streamlit
Master Class CESUMA · Tema 10: Ecosistemas en la nube y puesta en producción de sistemas de IA
Autor: Dr. Miguel Angel Domínguez González

Ejecución local:   streamlit run app.py
Despliegue:        Streamlit Community Cloud (GitHub)  ·  AWS EC2/ECS con Docker
"""

import io
import logging
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# 1. Configuración general
# ---------------------------------------------------------------------------
APP_VERSION = "1.0.0"                      # versión de la aplicación (versionado)
MODELOS_DISPONIBLES = {                    # versión del modelo -> archivo de pesos
    "YOLO26n (rápido, recomendado)": "yolo26n.pt",
    "YOLO11n (alternativo)": "yolo11n.pt",
}
CARPETA_MODELOS = Path(__file__).parent / "models"
LADO_MAXIMO = 1280                         # px: limita el costo de inferencia
TAMANO_MAX_MB = 10                         # coincide con .streamlit/config.toml

# Monitorización básica: los logs aparecen en "Manage app" (Streamlit Cloud)
# o en CloudWatch / `docker logs` (AWS).
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("yolo-app")

st.set_page_config(page_title="Detector de objetos YOLO", page_icon="🔎", layout="wide")


# ---------------------------------------------------------------------------
# 2. Carga del modelo UNA sola vez por proceso (servidor de modelo "ligero")
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando modelo YOLO…")
def cargar_modelo(nombre_pesos: str) -> YOLO:
    """Si el archivo de pesos está versionado en ./models lo usa;
    si no, Ultralytics lo descarga automáticamente la primera vez."""
    ruta_local = CARPETA_MODELOS / nombre_pesos
    origen = str(ruta_local) if ruta_local.exists() else nombre_pesos
    log.info("Cargando modelo: %s", origen)
    return YOLO(origen)


def preparar_imagen(archivo) -> Image.Image:
    """Mismo preprocesamiento en desarrollo y en producción (evita training-serving skew)."""
    img = Image.open(archivo)
    img = ImageOps.exif_transpose(img)     # respeta la orientación de fotos de celular
    img = img.convert("RGB")
    img.thumbnail((LADO_MAXIMO, LADO_MAXIMO))
    return img


def detectar(modelo: YOLO, img: Image.Image, conf: float, iou: float):
    t0 = time.perf_counter()
    resultado = modelo.predict(img, conf=conf, iou=iou, imgsz=640, verbose=False)[0]
    latencia_ms = (time.perf_counter() - t0) * 1000

    filas = []
    for caja in resultado.boxes:
        x1, y1, x2, y2 = (round(v) for v in caja.xyxy[0].tolist())
        filas.append({
            "Objeto": resultado.names[int(caja.cls)],
            "Confianza": round(float(caja.conf), 3),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        })
    tabla = pd.DataFrame(filas, columns=["Objeto", "Confianza", "x1", "y1", "x2", "y2"])
    anotada = Image.fromarray(resultado.plot()[:, :, ::-1])   # BGR -> RGB
    return anotada, tabla, latencia_ms


# ---------------------------------------------------------------------------
# 3. Interfaz
# ---------------------------------------------------------------------------
st.title("🔎 Detector de objetos con YOLO")
st.caption("Sube una imagen y el modelo identificará los objetos que contiene "
           "(80 clases del conjunto COCO: personas, autos, perros, sillas, laptops…).")

with st.sidebar:
    st.header("⚙️ Parámetros")
    etiqueta_modelo = st.selectbox("Modelo", list(MODELOS_DISPONIBLES))
    conf = st.slider("Umbral de confianza", 0.05, 0.95, 0.25, 0.05,
                     help="Detecciones con confianza menor se descartan.")
    iou = st.slider("Umbral IoU", 0.10, 0.90, 0.45, 0.05,
                    help="Controla la supresión de cajas duplicadas.")
    st.divider()
    st.caption(f"App v{APP_VERSION} · Modelo: {MODELOS_DISPONIBLES[etiqueta_modelo]}")

modelo = cargar_modelo(MODELOS_DISPONIBLES[etiqueta_modelo])

archivo = st.file_uploader("Selecciona una imagen (JPG, PNG o WEBP)",
                           type=["jpg", "jpeg", "png", "webp"])

if archivo is None:
    st.info("👆 Sube una imagen para comenzar.")
    st.stop()

if archivo.size > TAMANO_MAX_MB * 1024 * 1024:
    st.error(f"La imagen supera {TAMANO_MAX_MB} MB.")
    st.stop()

try:
    imagen = preparar_imagen(archivo)
except Exception as e:                     # archivo corrupto o formato no válido
    log.warning("Imagen inválida: %s", e)
    st.error("No se pudo leer la imagen. Intenta con otro archivo.")
    st.stop()

with st.spinner("Detectando objetos…"):
    anotada, tabla, latencia_ms = detectar(modelo, imagen, conf, iou)

log.info("inferencia modelo=%s objetos=%d latencia_ms=%.1f",
         MODELOS_DISPONIBLES[etiqueta_modelo], len(tabla), latencia_ms)

col1, col2 = st.columns(2)
col1.subheader("Imagen original")
col1.image(imagen, width="stretch")
col2.subheader("Objetos detectados")
col2.image(anotada, width="stretch")

m1, m2, m3 = st.columns(3)
m1.metric("Objetos detectados", len(tabla))
m2.metric("Clases distintas", tabla["Objeto"].nunique())
m3.metric("Latencia de inferencia", f"{latencia_ms:.0f} ms")

if tabla.empty:
    st.warning("No se detectaron objetos. Prueba bajando el umbral de confianza.")
else:
    c1, c2 = st.columns([1, 2])
    conteo = tabla["Objeto"].value_counts().rename_axis("Objeto").reset_index(name="Cantidad")
    c1.subheader("Resumen")
    c1.dataframe(conteo, hide_index=True, width="stretch")
    c2.subheader("Detalle de detecciones")
    c2.dataframe(tabla, hide_index=True, width="stretch")

buffer = io.BytesIO()
anotada.save(buffer, format="PNG")
st.download_button("⬇️ Descargar imagen anotada", buffer.getvalue(),
                   file_name="deteccion_yolo.png", mime="image/png")
