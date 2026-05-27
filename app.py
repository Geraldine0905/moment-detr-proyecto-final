import streamlit as st
import subprocess
import sys
import os
import json
import re

st.set_page_config(
    page_title="Moment-DETR Demo",
    layout="centered"
)

st.title("Detección de momentos en video con Moment-DETR")

st.write(
    "Sube un video, escribe una consulta en inglés y el modelo localizará "
    "el momento más relevante dentro del video."
)

video_file = st.file_uploader("Sube un video", type=["mp4", "mov", "avi"])
query = st.text_input("Consulta en inglés", "showing beauty product")

video_path = None

if video_file is not None:
    os.makedirs("videos", exist_ok=True)
    video_path = os.path.join("videos", video_file.name)

    with open(video_path, "wb") as f:
        f.write(video_file.read())

    st.subheader("Video cargado")
    st.video(video_path)

if st.button("Ejecutar inferencia"):
    if video_file is None:
        st.warning("Primero sube un video.")
    elif query.strip() == "":
        st.warning("Escribe una consulta en inglés.")
    else:
        query_path = "run_on_video/example/queries.jsonl"

        data = {
            "query": query,
            "video_path": video_path,
            "relevant_windows": [[0, 1]],
            "saliency_scores": [[0, 0, 0]]
        }

        with open(query_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")

        with st.spinner("Ejecutando Moment-DETR en CPU... puede tardar un poco"):
            result = subprocess.run(
                [sys.executable, "-m", "run_on_video.run"],
                capture_output=True,
                text=True,
                timeout=300
            )

        salida = result.stdout

        st.subheader("Resultado del modelo")
        st.code(salida)

        patron_momento = r"Momento detectado:\s*([\d\.]+)s\s*-\s*([\d\.]+)s"
        match_momento = re.search(patron_momento, salida)

        patron_confianza = r"Confianza:\s*([\d\.]+)"
        match_confianza = re.search(patron_confianza, salida)

        if match_momento:
            inicio = float(match_momento.group(1))
            fin = float(match_momento.group(2))

            st.success(f"Momento detectado entre {inicio:.2f}s y {fin:.2f}s")

            os.makedirs("outputs", exist_ok=True)
            clip_path = "outputs/momento_detectado.mp4"

            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i", video_path,
                    "-ss", str(inicio),
                    "-to", str(fin),
                    "-c", "copy",
                    clip_path
                ],
                capture_output=True,
                text=True
            )

            st.subheader("Fragmento detectado por el modelo")
            st.video(clip_path)

        else:
            st.warning("No se pudo extraer automáticamente el intervalo detectado.")

        if match_confianza:
            score = float(match_confianza.group(1))
            st.metric("Confianza", f"{score:.4f}")
        else:
            st.warning("No se pudo extraer automáticamente la confianza.")

        if result.stderr and "Traceback" in result.stderr:
            st.subheader("Errores")
            st.code(result.stderr)