import streamlit as st
import librosa
import librosa.display
import numpy as np
import io
import matplotlib.pyplot as plt

# Título de la aplicación
st.title("🎶 Analizador Musical")

# Subtítulo con una breve descripción
st.markdown("""
    Sube un archivo de audio (.wav o .mp3) para analizar su **BPM** y **Tonalidad**.
""")

# Widget para subir archivos
uploaded_file = st.file_uploader(
    "Elige un archivo de audio", 
    type=['mp3', 'wav'],
    help="Sube un archivo de música para el análisis. Se recomiendan archivos de menor duración para un análisis más rápido."
)

# Verifica si se ha subido un archivo
if uploaded_file is not None:
    # Muestra un mensaje de éxito
    st.success("Archivo subido con éxito. Analizando...")

    # Usa un spinner mientras se procesa el archivo
    with st.spinner('Cargando y procesando el archivo de audio...'):
        # Leer el archivo de audio cargado en bytes
        audio_bytes = uploaded_file.read()
        
        try:
            # Cargar el archivo de audio usando librosa
            # La función `load` necesita una ruta o un objeto similar a un archivo, 
            # por lo que usamos `io.BytesIO`
            y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)

            # --- Análisis del BPM (Beats Per Minute) ---
            # Se usa `librosa.beat.tempo` para estimar el tempo del audio.
            # El array resultante a menudo contiene múltiples estimaciones; se toma la primera.
            tempo, _ = librosa.beat.tempo(y=y, sr=sr, start_bpm=80, end_bpm=200)
            bpm = tempo[0]

            # --- Análisis de la Tonalidad ---
            # 1. Se calcula el espectrograma de croma (chroma)
            # El croma representa la intensidad de cada una de las 12 notas (C, C#, D, etc.)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            
            # 2. Se estima la tonalidad (key) a partir del espectrograma de croma.
            # `librosa.key_to_notes` convierte el índice de la nota a su nombre.
            key_index = np.argmax(np.mean(chroma, axis=1))
            key = librosa.key_to_note(key_index)

            # --- Mostrar los resultados en la interfaz ---
            st.header("Resultados del Análisis")
            
            # Usa `st.metric` para mostrar el BPM de manera destacada
            st.metric("BPM (Pulsos por Minuto)", f"{int(bpm)}", "Estimado")
            
            # Muestra la tonalidad
            st.metric("Tonalidad (Key)", key)

            # --- Visualizaciones ---
            st.header("Visualizaciones del Audio")
            
            # 1. Gráfico de la forma de onda (waveform)
            # Muestra la amplitud del audio a lo largo del tiempo.
            fig_waveform, ax_waveform = plt.subplots(figsize=(10, 4))
            librosa.display.waveshow(y, sr=sr, ax=ax_waveform, x_axis='time')
            ax_waveform.set_title('Forma de Onda del Audio')
            ax_waveform.set_xlabel('Tiempo (s)')
            ax_waveform.set_ylabel('Amplitud')
            st.pyplot(fig_waveform)
            plt.close(fig_waveform) # Cierra la figura para evitar problemas con Streamlit

            # 2. Gráfico del Cromagrama
            # Muestra la intensidad de las 12 clases de tonos (C, C#, D, etc.) a lo largo del tiempo.
            fig_chroma, ax_chroma = plt.subplots(figsize=(10, 4))
            img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', sr=sr, ax=ax_chroma)
            ax_chroma.set_title('Cromagrama (Intensidad de Notas)')
            ax_chroma.set_xlabel('Tiempo (s)')
            ax_chroma.set_ylabel('Notas')
            st.pyplot(fig_chroma)
            plt.close(fig_chroma)

        except Exception as e:
            # Manejo de errores en caso de que el análisis falle
            st.error(f"Ocurrió un error al procesar el archivo: {e}")
            st.info("Asegúrate de que el archivo no esté corrupto y sea un formato de audio válido (.wav o .mp3).")

# Sección de pie de página
st.markdown("---")
st.markdown("Creado con `streamlit` y `librosa`.")
