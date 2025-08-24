# 🎶 Analizador Musical (Lite) — v1.2
Para Streamlit Cloud, sin librosa/numba. Formatos: mp3/wav/m4a/ogg con pydub+ffmpeg.

## Despliegue
1) Sube estos archivos a un repo (raíz).  
2) Streamlit Cloud → New app → `app.py` → Python 3.11.  
3) **packages.txt** instala `ffmpeg` y `libsndfile1`.

## Funciones
- **BPM**: spectral flux + autocorrelación (60–180 bpm, con ajuste half/double).  
- **Tonalidad**: cromagrama 12-TET + perfiles Krumhansl (mayor/menor).  
- **Notas (opcional)**: YIN-lite monofónico con segmentación.

## Consejos
- Limita **Máx. duración** a 60–90s para análisis rápido.  
- Notas: usa pistas **monofónicas** (voz, bajo, lead).

