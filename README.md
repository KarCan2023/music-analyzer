# 🎶 Analizador Musical (Balanced) — v1.2.2
Equilibrado para Streamlit Cloud: **BPM + Tonalidad + Recomendador**.  
Sin extracción de **notas** (lo más pesado) para evitar problemas de instalación/tiempo.

## Deploy (GitHub → Streamlit Cloud)
1) Sube estos archivos al **raíz** del repo.  
2) En Streamlit Cloud: New app → `app.py`.  
3) **Runtime**: Python 3.13 (o 3.11). `requirements.txt` incluye **pyaudioop** (necesario para pydub en 3.13).  
4) `packages.txt` instala **ffmpeg** y **libsndfile1**.

## Qué hace
- **BPM**: flux + autocorrelación (ajuste half/double).  
- **Tonalidad**: cromagrama 12‑TET + Krumhansl (mayor/menor).  
- **Recomendador**: sugiere géneros por rango BPM y modo (mayor/menor).

## Tips
- Usa **SR 22.05 kHz** y **60–90s** de audio para ir fluido.  
- Si el BPM sale en doble/mitad, interpreta según el género (ej. 70 ↔ 140).

