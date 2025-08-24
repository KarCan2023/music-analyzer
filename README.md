# 🎵 Streamlit: Analizador de Tempo, Tonalidad y Notas (HPSS) — v1.1.1 Safe Mode

Paquete listo para GitHub + Streamlit Cloud, con **modo seguro**, excepciones controladas y dependencias del sistema (`ffmpeg`, `libsndfile1`).

## Despliegue
1) Sube todo el contenido al **raíz** de tu repo en GitHub.  
2) En **Streamlit Cloud**: New app → repo → `app.py`. En Settings, usa **Python 3.11**.  
3) Este repo incluye `packages.txt` con `ffmpeg` y `libsndfile1` (necesario para decodificar audio).

## Si ves “Oh no. Error running app.”
- Activa **🛡️ Modo Seguro** (en el sidebar).
- Baja **Máx. duración** (ej. 60–90s) y **Sample rate** (22.05 kHz).
- Desactiva **HPSS** y **Notas** y ve activando uno por vez.
- Abre **Manage app → Logs** y mira el *Traceback* (palabras clave típicas: `libsndfile`, `ffmpeg`, `numba`, `MemoryError`).

## Instalar local
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Notas
- Para mejores notas, usa **señal Harmónica (HPSS)**.
- Para stems vocal/instrumental de calidad de estudio, montar Spleeter/Demucs en servicio externo y conectar por API.
