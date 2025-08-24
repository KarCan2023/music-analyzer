# 🎶 Analizador Musical (Ultralite) — v1.2.1
**Cero SciPy y cero Librosa.** Compatible con Python **3.11–3.13**.  
Soporta MP3/M4A/OGG/WAV con **pydub + ffmpeg**. Para Python 3.13 incluimos **pyaudioop** (sustituye al módulo stdlib removido `audioop`).

## Despliegue (GitHub → Streamlit Cloud)
1. Sube todos los archivos al **raíz** de tu repo.
2. En Streamlit Cloud: New app → `app.py`. Puedes usar **Python 3.13** o 3.11.
3. Este repo incluye `packages.txt` para instalar **ffmpeg** y **libsndfile1** (decodificación).

## Funciones incluidas
- **BPM**: STFT (NumPy) + **Spectral Flux** + **Autocorrelación** (con ajuste half/double).
- **Tonalidad**: cromagrama 12‑TET + **Krumhansl** (mayor/menor).
- **Notas (opcional)**: **YIN‑lite** monofónico + segmentación.

## Recomendaciones
- Máx. duración 60–90s para ir rápido.
- Notas: usa pistas **monofónicas** para mejor precisión.
