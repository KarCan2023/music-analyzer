import tempfile
import io
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf

# =============================
# Configuración general UI
# =============================
st.set_page_config(
    page_title="Analizador de Tempo, Tonalidad y Notas",
    page_icon="🎵",
    layout="wide",
)

CUSTOM_CSS = """
<style>
section.main > div {max-width: 1200px;}
.block-container {padding-top: 1.2rem;}
.metric-card {background: #0E1117; border: 1px solid #2B2F3A; border-radius: 14px; padding: 16px;}
.badge {display:inline-block; padding:6px 10px; border-radius: 999px; border:1px solid #2B2F3A; margin:4px 6px 0 0; font-size:0.85rem}
.small {opacity:.8;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("🎵 Analizador de Tempo, Tonalidad y Notas")
st.caption("Sube tu audio y obtén BPM, tonalidad y notas. Incluye **HPSS**. Versión segura/rápida v1.1.1.")

# =============================
# Utilidades musicales
# =============================
PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

@dataclass
class StyleRule:
    name: str
    bpm_min: int
    bpm_max: int
    key_mode: str  # "any" | "major" | "minor" | "list"
    keys: Optional[List[str]] = None

SEED_STYLES = [
    StyleRule("Reggaetón", 85, 100, "minor"),
    StyleRule("Trap", 70, 85, "minor"),
    StyleRule("Hip-Hop / Boom Bap", 85, 98, "minor"),
    StyleRule("Afrobeats", 95, 115, "any"),
    StyleRule("Pop", 92, 116, "any"),
    StyleRule("House", 120, 128, "any"),
    StyleRule("Techno", 125, 135, "minor"),
    StyleRule("Trance", 130, 142, "minor"),
    StyleRule("Dubstep", 138, 142, "minor"),
    StyleRule("Drum & Bass", 165, 180, "minor"),
    StyleRule("Dembow", 95, 105, "minor"),
    StyleRule("Bachata", 120, 136, "minor"),
    StyleRule("Merengue", 120, 160, "major"),
    StyleRule("Salsa", 180, 230, "major"),
    StyleRule("Lofi / Chillhop", 60, 90, "minor"),
    StyleRule("Rock", 100, 140, "any"),
]

# =============================
# Cache liviano (solo estilos)
# =============================
@st.cache_data(show_spinner=False)
def load_styles_from_json(file_bytes: Optional[bytes]):
    if not file_bytes:
        df = pd.DataFrame([r.__dict__ for r in SEED_STYLES])
        return df
    try:
        df = pd.read_json(io.BytesIO(file_bytes))
        if "keys" not in df.columns:
            df["keys"] = None
        return df
    except Exception as e:
        st.warning(f"No se pudo leer el JSON de estilos: {e}. Usando seed por defecto.")
        return pd.DataFrame([r.__dict__ for r in SEED_STYLES])

# =============================
# Procesamiento de audio
# =============================
def save_temp_uploaded_file(uploaded_file) -> str:
    suffix = ".mp3"
    n = uploaded_file.name.lower()
    if n.endswith(".wav"): suffix = ".wav"
    elif n.endswith(".m4a"): suffix = ".m4a"
    elif n.endswith(".ogg"): suffix = ".ogg"
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tf.write(uploaded_file.getbuffer())
    tf.flush()
    return tf.name

def load_audio(path: str, sr: int = 22050, mono: bool = True, max_seconds: Optional[int] = None):
    y, sr = librosa.load(path, sr=sr, mono=mono)
    if max_seconds is not None:
        y = y[: int(max_seconds * sr)]
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    return y, sr

def estimate_bpm(y: np.ndarray, sr: int) -> Tuple[float, np.ndarray]:
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units='frames')
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return float(tempo), beat_times

def chroma_mean(y: np.ndarray, sr: int) -> np.ndarray:
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma /= (chroma.max() + 1e-9)
    return np.mean(chroma, axis=1)

def estimate_key(y: np.ndarray, sr: int) -> Tuple[str, float]:
    chroma_vec = chroma_mean(y, sr)
    def roll(p, n): return np.roll(p, n)
    best_key, best_score = None, -np.inf
    scores = []
    for i, tonic in enumerate(PITCH_CLASSES):
        maj = np.corrcoef(chroma_vec, roll(MAJOR_PROFILE, i))[0, 1]
        minr = np.corrcoef(chroma_vec, roll(MINOR_PROFILE, i))[0, 1]
        scores.append((f"{tonic} major", maj))
        scores.append((f"{tonic} minor", minr))
        if maj > best_score: best_score, best_key = maj, f"{tonic} major"
        if minr > best_score: best_score, best_key = minr, f"{tonic} minor"
    scores.sort(key=lambda x: x[1], reverse=True)
    top1, top2 = scores[0][1], scores[1][1]
    return best_key, float(max(0.0, top1 - top2))

def extract_notes(y: np.ndarray, sr: int, fmin: str = 'C2', fmax: str = 'C7', frame_length: int = 2048, hop_length: int = 256, threshold_prob: float = 0.7) -> pd.DataFrame:
    fmin_hz = librosa.note_to_hz(fmin)
    fmax_hz = librosa.note_to_hz(fmax)
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=fmin_hz, fmax=fmax_hz, frame_length=frame_length, hop_length=hop_length
    )
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    is_note = (voiced_flag == True) & (voiced_prob >= threshold_prob) & (~np.isnan(f0))
    f0_clean = np.where(is_note, f0, np.nan)
    midi = librosa.hz_to_midi(f0_clean, hz=None)
    midi_round = np.rint(midi)
    note_names = np.array([librosa.midi_to_note(int(m), octave=True, unicode=False) if not np.isnan(m) else None for m in midi_round])
    segments = []
    current_note, start_time = None, None
    values_hz, values_midi = [], []
    for t, nn, hz, mm in zip(times, note_names, f0_clean, midi_round):
        if nn is None:
            if current_note is not None:
                end_time = t
                if values_hz:
                    segments.append({
                        "note": current_note,
                        "midi": float(np.nanmedian(values_midi)),
                        "freq_hz": float(np.nanmedian(values_hz)),
                        "start_s": float(start_time),
                        "end_s": float(end_time),
                        "duration_s": float(end_time - start_time)
                    })
                current_note, start_time, values_hz, values_midi = None, None, [], []
            continue
        if current_note is None:
            current_note, start_time = nn, t
            values_hz, values_midi = [hz], [mm]
        elif nn == current_note:
            values_hz.append(hz); values_midi.append(mm)
        else:
            end_time = t
            if values_hz:
                segments.append({
                    "note": current_note,
                    "midi": float(np.nanmedian(values_midi)),
                    "freq_hz": float(np.nanmedian(values_hz)),
                    "start_s": float(start_time),
                    "end_s": float(end_time),
                    "duration_s": float(end_time - start_time)
                })
            current_note, start_time = nn, t
            values_hz, values_midi = [hz], [mm]
    if current_note is not None and values_hz:
        end_time = times[-1]
        segments.append({
            "note": current_note,
            "midi": float(np.nanmedian(values_midi)),
            "freq_hz": float(np.nanmedian(values_hz)),
            "start_s": float(start_time),
            "end_s": float(end_time),
            "duration_s": float(end_time - start_time)
        })
    df = pd.DataFrame(segments)
    if not df.empty: df = df.sort_values("start_s").reset_index(drop=True)
    return df

# =============================
# HPSS
# =============================
def separate_hpss(y: np.ndarray, margin_h: float = 1.0, margin_p: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    y_harm, y_perc = librosa.effects.hpss(y, margin=(margin_h, margin_p))
    for arr in (y_harm, y_perc):
        if np.max(np.abs(arr)) > 0: arr /= np.max(np.abs(arr))
    return y_harm, y_perc

# =============================
# Recomendador
# =============================
def key_to_mode(key_str: str) -> str: return "major" if key_str.lower().endswith("major") else "minor"
def normalize_key_str(key_str: str) -> str: return key_str.replace("\n", " ").strip()

def recommend_styles(bpm: float, key_str: str, styles_df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    key_str = normalize_key_str(key_str); mode = key_to_mode(key_str)
    def bpm_score(row):
        lo, hi = row["bpm_min"], row["bpm_max"]
        diff = 0 if lo <= bpm <= hi else min(abs(bpm-lo), abs(bpm-hi))
        return float(max(0.0, 1.0 - (diff / 30.0)))
    def key_score(row):
        kmode = str(row.get("key_mode","any")).lower()
        if kmode == "any": return 1.0
        if kmode in ("major","minor"): return 1.0 if kmode == mode else 0.4
        if kmode == "list":
            keys = row.get("keys")
            if isinstance(keys, list) and key_str in keys: return 1.0
            return 0.6 if any(str(k).lower().endswith(mode) for k in (keys or [])) else 0.2
        return 0.6
    out = styles_df.copy()
    out["bpm_score"] = out.apply(bpm_score, axis=1)
    out["key_score"] = out.apply(key_score, axis=1)
    out["score"] = 0.7*out["bpm_score"] + 0.3*out["key_score"]
    return out.sort_values("score", ascending=False).head(top_k)[["name","bpm_min","bpm_max","key_mode","keys","score","bpm_score","key_score"]]

# =============================
# Sidebar (controles)
# =============================
with st.sidebar:
    st.header("⚙️ Configuración")
    safe_mode = st.checkbox("🛡️ Modo Seguro (rápido)", value=True, help="Desactiva HPSS/Notas/Gráficos para evitar errores de tiempo/memoria en Cloud.")
    max_seconds = st.slider("Máx. duración a analizar (seg)", 10, 300, 90, 10, help="Recorta para acelerar.")
    sr = st.selectbox("Sample rate (Hz)", [16000, 22050, 32000, 44100], index=1)
    show_plots = st.checkbox("Mostrar visualizaciones", value=not safe_mode)
    st.markdown("---")
    st.subheader("🪄 Stems (HPSS)")
    use_hpss = st.checkbox("Separar en Harmónica/Percusiva", value=not safe_mode)
    margin_h = st.slider("Margen armónico", 0.5, 5.0, 1.0, 0.1)
    margin_p = st.slider("Margen percusivo", 0.5, 5.0, 1.0, 0.1)
    st.markdown("---")
    st.subheader("🎼 Notas (PYIN)")
    extract_notes_flag = st.checkbox("Extraer notas (monofónico)", value=not safe_mode)
    pyin_prob = st.slider("Umbral PYIN", 0.0, 1.0, 0.70, 0.05)
    st.markdown("---")
    st.subheader("📚 Base de estilos")
    styles_json = st.file_uploader("Cargar JSON de estilos (opcional)", type=["json"])

styles_df = load_styles_from_json(styles_json.read() if styles_json is not None else None)
with st.expander("Editar estilos (opcional)"):
    st.write("Ajusta rangos y modos. Con 'list' agrega claves exactas p.ej. 'A minor'.")
    styles_df = st.data_editor(styles_df, num_rows="dynamic")

# =============================
# Subida y análisis
# =============================
uploaded = st.file_uploader("Sube tu audio (mp3/wav/m4a/ogg)", type=["mp3","wav","m4a","ogg"], accept_multiple_files=False)

if uploaded:
    if uploaded.size and uploaded.size > 30 * 1024 * 1024:
        st.warning("Archivo grande (>30MB). Se recomienda usar un recorte o reducir 'Máx. duración'.")

    try:
        temp_path = save_temp_uploaded_file(uploaded)
        y, sr_eff = load_audio(temp_path, sr=sr, mono=True, max_seconds=max_seconds)
    except Exception as e:
        st.exception(e)
        st.stop()

    col1, col2 = st.columns([1, 1])
    with col1: st.audio(uploaded)
    with col2:
        st.write("**Resumen del archivo**")
        st.write({"sample_rate": sr_eff, "duración_s": round(len(y)/sr_eff, 2), "frames": len(y)})

    stems_dict = {"Original": y}
    if use_hpss:
        try:
            with st.spinner("Separando (HPSS)…"):
                y_harm, y_perc = separate_hpss(y)
                stems_dict["Harmónica"] = y_harm
                stems_dict["Percusiva"] = y_perc
        except Exception as e:
            st.warning(f"HPSS falló: {e}. Continuando con señal original.")
            use_hpss = False

    sel = st.radio("Señal para analizar", list(stems_dict.keys()), index=0, horizontal=True)
    y_use = stems_dict[sel]

    with st.expander("Descargar stems (WAV)"):
        for name, sig in stems_dict.items():
            try:
                buf = io.BytesIO(); sf.write(buf, sig, sr_eff, format='WAV')
                st.download_button(f"Descargar {name}.wav", data=buf.getvalue(), file_name=f"{name.lower()}.wav", mime="audio/wav")
            except Exception as e:
                st.warning(f"No se pudo exportar {name}: {e}")

    try:
        with st.spinner("Analizando tempo y tonalidad..."):
            bpm, beat_times = estimate_bpm(y_use, sr_eff)
            key_str, key_conf = estimate_key(y_use, sr_eff)
    except Exception as e:
        st.exception(e)
        st.stop()

    m1, m2 = st.columns(2)
    with m1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("BPM (estimado)", f"{bpm:.1f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Tonalidad (clave)", key_str)
        st.caption(f"Confianza relativa: {key_conf:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("🎯 Estilos recomendados")
    recs = recommend_styles(bpm, key_str, styles_df)
    if recs.empty: st.info("Sin recomendaciones con la configuración actual.")
    else:
        for _, row in recs.iterrows():
            st.markdown(f"<span class='badge'>**{row['name']}** · {int(row['bpm_min'])}-{int(row['bpm_max'])} BPM · modo: {row['key_mode']} · score: {row['score']:.2f}</span>", unsafe_allow_html=True)

    if show_plots:
        st.markdown("---")
        st.subheader("📊 Visualizaciones")
        try:
            fig, ax = plt.subplots(figsize=(8, 3))
            librosa.display.waveshow(y_use, sr=sr_eff, ax=ax)
            for bt in beat_times: ax.axvline(bt, alpha=0.3, linestyle='--')
            ax.set_xlabel("Tiempo (s)"); ax.set_ylabel("Amplitud")
            st.pyplot(fig, use_container_width=True); plt.close(fig)

            chroma = librosa.feature.chroma_cqt(y=y_use, sr=sr_eff)
            fig2, ax2 = plt.subplots(figsize=(8, 3))
            img = librosa.display.specshow(chroma, x_axis='time', y_axis='chroma', cmap='magma', ax=ax2)
            ax2.set_title('Chroma CQT')
            fig2.colorbar(img, ax=ax2, format='%+0.2f')
            st.pyplot(fig2, use_container_width=True); plt.close(fig2)
        except Exception as e:
            st.warning(f"No se pudieron generar gráficos: {e}")

    if extract_notes_flag:
        st.markdown("---")
        st.subheader("🎼 Notas detectadas (monofónico)")
        try:
            with st.spinner("Extrayendo notas con PYIN…"):
                notes_df = extract_notes(y_use, sr_eff, threshold_prob=pyin_prob)
            if notes_df.empty:
                st.info("No se detectaron notas confiables. Prueba con la señal Harmónica o baja el umbral.")
            else:
                st.dataframe(notes_df, use_container_width=True)
                csv = notes_df.to_csv(index=False).encode('utf-8')
                st.download_button("Descargar notas CSV", data=csv, file_name="notas_detectadas.csv", mime="text/csv")
                try:
                    fig3, ax3 = plt.subplots(figsize=(10, 3))
                    for _, r in notes_df.iterrows():
                        ax3.hlines(r["midi"], r["start_s"], r["end_s"], linewidth=4)
                    ax3.set_xlabel("Tiempo (s)"); ax3.set_ylabel("MIDI note")
                    st.pyplot(fig3, use_container_width=True); plt.close(fig3)
                except Exception as e:
                    st.warning(f"No se pudo dibujar el piano-roll: {e}")
        except Exception as e:
            st.warning(f"PYIN falló: {e}. Desactiva notas o usa Modo Seguro.")
            st.stop()

st.markdown("---")
st.caption("v1.1.1 — 'Safe mode'. Si Streamlit Cloud muestra 'Oh no', baja duración, usa 22.05 kHz, desactiva HPSS/Notas, y vuelve a activar por partes.")
