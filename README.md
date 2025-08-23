# 🎵 Streamlit: Analizador de Tempo, Tonalidad y Notas (con HPSS)

App para productores musicales: sube un audio (mp3/wav/m4a/ogg) y obtén **BPM**, **tonalidad (mayor/menor)** y **notas** (cifrado americano, vía PYIN). Incluye **separación de stems (HPSS)** para mejorar la precisión.

## ✨ Funcionalidades
- Estimación de **tempo (BPM)** y marcadores de **beats**.
- Detección de **clave** por cromas (Krumhansl).
- **PYIN** para extraer **notas monofónicas** (ideal en pista harmónica o voces/bajo).
- **HPSS** (harmónica/percusiva) integrada. Descarga cada stem en WAV.
- **Recomendador de estilos** por BPM/clave con BD editable (JSON o editor en la app).
- Visualizaciones: onda+beats, cromagrama CQT, piano‑roll simple de notas.

## 📦 Estructura
```
streamlit-bpm-key-notes-v1.1/
├─ app.py
├─ requirements.txt
├─ styles_seed.json
├─ packages.txt          # instala ffmpeg (Streamlit Cloud)
└─ .streamlit/
   └─ config.toml        # tema opcional
```

## 🚀 Despliegue (GitHub → Streamlit Cloud)
1. Crea un repo y sube estos archivos (raíz del repo).
2. En **Streamlit Cloud**: *New app* → conecta tu GitHub → selecciona el repo/rama → `app.py`.
3. **Python**: 3.11 recomendado.
4. **packages.txt** ya incluye `ffmpeg` (necesario para algunos formatos).

## ▶️ Ejecución local
```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
streamlit run app.py
```

Si tu entorno no decodifica MP3/M4A, instala `ffmpeg` o convierte a WAV.

## 🛠 Uso recomendado
- Para **notas**: analiza la **señal Harmónica (HPSS)** y ajusta el **umbral PYIN** si hay ruido.
- En géneros con **double-time** (70 ↔ 140) interpreta el BPM con criterio musical.
- Ajusta y guarda tu **BD de estilos** en `styles_seed.json` y cárgala desde el sidebar.

## 🧩 Extensiones futuras
- Exportar **MIDI** de las notas detectadas.
- Detección de **downbeat** y clave por secciones (modulaciones).
- Integración con **Spleeter/Demucs** vía servicio externo (API).

---
**Créditos**: Librosa/Numba/SoundFile/Matplotlib/Streamlit.
