import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import soundfile as sf
from pydub import AudioSegment  # ffmpeg backend

st.set_page_config(page_title="Analizador Musical (Balanced)", page_icon="🎶", layout="wide")
st.title("🎶 Analizador de BPM, Tonalidad y Recomendaciones (Balanced)")
st.caption("Edición equilibrada: BPM + Tonalidad + Recomendador de estilos. Sin extracción de notas para máxima estabilidad en Streamlit Cloud.")

# ---------- Audio helpers ----------
def resample_linear(y: np.ndarray, sr_orig: int, sr_target: int) -> np.ndarray:
    if sr_orig == sr_target or y.size == 0:
        return y.astype(np.float32)
    x_old = np.linspace(0.0, 1.0, num=len(y), endpoint=False)
    n_new = int(round(len(y) * sr_target / sr_orig))
    x_new = np.linspace(0.0, 1.0, num=n_new, endpoint=False)
    return np.interp(x_new, x_old, y).astype(np.float32)

def read_audio_any(file, target_sr=22050, mono=True):
    # Primary: pydub (ffmpeg)
    try:
        seg = AudioSegment.from_file(file)
        if mono and seg.channels > 1: seg = seg.set_channels(1)
        if seg.frame_rate != target_sr: seg = seg.set_frame_rate(target_sr)
        samples = seg.get_array_of_samples()
        y = np.array(samples).astype(np.float32)
        # normalize by sample width
        if seg.sample_width == 1:
            y = (y - 128.0) / 128.0
        elif seg.sample_width == 2:
            y = y / 32768.0
        elif seg.sample_width == 3:
            y = y / (2**23)
        elif seg.sample_width == 4:
            y = y / (2**31)
        if seg.channels == 2 and mono:
            y = y.reshape((-1, 2)).mean(axis=1)
        return y, seg.frame_rate
    except Exception as e:
        # Fallback: wav/ogg via soundfile
        try:
            data, sr = sf.read(file, dtype="float32", always_2d=False)
            if mono and data.ndim > 1:
                data = data.mean(axis=1)
            if sr != target_sr:
                data = resample_linear(data, sr, target_sr)
                sr = target_sr
            return data.astype(np.float32), sr
        except Exception as e2:
            raise RuntimeError(f"No se pudo leer el audio. Revisa ffmpeg/libsndfile. Detalle: {e} | {e2}")

def normalize(y: np.ndarray) -> np.ndarray:
    m = float(np.max(np.abs(y))) if y.size else 0.0
    return (y / m).astype(np.float32) if m > 0 else y.astype(np.float32)

# ---------- STFT (NumPy) ----------
def stft_mag(y: np.ndarray, sr: int, n_fft=2048, hop=512, win="hann"):
    if y.size < n_fft:
        return np.fft.rfftfreq(n_fft, 1/sr), np.array([]), np.zeros((n_fft//2+1,0), dtype=np.float32)
    window = np.hanning(n_fft).astype(np.float32) if win=="hann" else np.ones(n_fft, dtype=np.float32)
    n_frames = 1 + (len(y) - n_fft) // hop
    mags = []
    for i in range(n_frames):
        start = i*hop
        frame = y[start:start+n_fft] * window
        spec = np.fft.rfft(frame, n_fft)
        mags.append(np.abs(spec).astype(np.float32))
    mag = np.stack(mags, axis=1) if mags else np.zeros((n_fft//2+1,0), dtype=np.float32)
    freqs = np.fft.rfftfreq(n_fft, 1/sr)
    times = (np.arange(n_frames)*hop + (n_fft/2)) / sr
    return freqs, times, mag

# ---------- BPM via flux + ACF ----------
def spectral_flux(mag: np.ndarray):
    if mag.shape[1] < 2:
        return np.zeros(0, dtype=np.float32)
    diff = np.diff(mag, axis=1)
    diff[diff < 0] = 0.0
    flux = diff.sum(axis=0).astype(np.float32)
    if flux.max() > 0: flux /= flux.max()
    return flux

def bpm_from_flux_acf(flux: np.ndarray, sr: int, hop: int, bpm_min=60, bpm_max=180):
    if flux.size < 4:
        return 0.0, 0.0
    x = flux - flux.mean()
    acf = np.correlate(x, x, mode="full")[len(x)-1:]
    lag_min = int(round((60.0/bpm_max) * sr / hop))
    lag_max = int(round((60.0/bpm_min) * sr / hop))
    lag_min = max(lag_min, 1)
    lag_max = min(lag_max, len(acf)-1)
    if lag_max <= lag_min:
        return 0.0, 0.0
    lag = int(np.argmax(acf[lag_min:lag_max]) + lag_min)
    bpm = 60.0 * sr / (lag * hop)
    # adjust half/double
    if bpm < 75: bpm *= 2.0
    if bpm > 165: bpm /= 2.0
    period_s = 60.0 / max(bpm, 1e-6)
    return float(bpm), float(period_s)

# ---------- Key detection ----------
PITCH_CLASSES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
MAJOR_PROFILE = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88], dtype=np.float32)
MINOR_PROFILE = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17], dtype=np.float32)

def chromagram(mag: np.ndarray, freqs: np.ndarray, fmin=50.0):
    chroma = np.zeros((12, mag.shape[1]), dtype=np.float32)
    if mag.size == 0 or freqs.size == 0:
        return chroma
    valid = freqs >= fmin
    freqs_v = freqs[valid]
    mags_v = mag[valid, :]
    if freqs_v.size == 0:
        return chroma
    midi = 69 + 12*np.log2(freqs_v/440.0)
    pcs = (np.rint(midi) % 12).astype(int)
    for i in range(len(freqs_v)):
        chroma[pcs[i], :] += mags_v[i, :]
    s = chroma.sum(axis=0, keepdims=True) + 1e-9
    chroma = chroma / s
    return chroma

def estimate_key_from_chroma(chroma_mat: np.ndarray):
    if chroma_mat.size == 0 or chroma_mat.shape[1] == 0:
        return "Unknown", 0.0
    cm = chroma_mat.mean(axis=1)
    best_key, best_score = None, -1e9
    scores = []
    for i, tonic in enumerate(PITCH_CLASSES):
        maj = np.corrcoef(cm, np.roll(MAJOR_PROFILE, i))[0,1]
        mnr = np.corrcoef(cm, np.roll(MINOR_PROFILE, i))[0,1]
        scores.append((f"{tonic} major", maj))
        scores.append((f"{tonic} minor", mnr))
        if maj > best_score: best_score, best_key = maj, f"{tonic} major"
        if mnr > best_score: best_score, best_key = mnr, f"{tonic} minor"
    scores.sort(key=lambda x: x[1], reverse=True)
    conf = float(max(0.0, scores[0][1] - scores[1][1]))
    return best_key, conf

def mode_from_key(key_str: str) -> str:
    return "major" if key_str.lower().endswith("major") else "minor"

# ---------- Recomendador sencillo ----------
SEED_STYLES = [
    {"name":"Reggaetón","bpm_min":85,"bpm_max":100,"key_mode":"minor"},
    {"name":"Trap","bpm_min":70,"bpm_max":85,"key_mode":"minor"},
    {"name":"Hip-Hop / Boom Bap","bpm_min":85,"bpm_max":98,"key_mode":"minor"},
    {"name":"Afrobeats","bpm_min":95,"bpm_max":115,"key_mode":"any"},
    {"name":"Pop","bpm_min":92,"bpm_max":116,"key_mode":"any"},
    {"name":"House","bpm_min":120,"bpm_max":128,"key_mode":"any"},
    {"name":"Techno","bpm_min":125,"bpm_max":135,"key_mode":"minor"},
    {"name":"Trance","bpm_min":130,"bpm_max":142,"key_mode":"minor"},
    {"name":"Dubstep","bpm_min":138,"bpm_max":142,"key_mode":"minor"},
    {"name":"Drum & Bass","bpm_min":165,"bpm_max":180,"key_mode":"minor"},
    {"name":"Dembow","bpm_min":95,"bpm_max":105,"key_mode":"minor"},
    {"name":"Bachata","bpm_min":120,"bpm_max":136,"key_mode":"minor"},
    {"name":"Merengue","bpm_min":120,"bpm_max":160,"key_mode":"major"},
    {"name":"Salsa","bpm_min":180,"bpm_max":230,"key_mode":"major"},
    {"name":"Lofi / Chillhop","bpm_min":60,"bpm_max":90,"key_mode":"minor"},
    {"name":"Rock","bpm_min":100,"bpm_max":140,"key_mode":"any"}
]
def recommend_styles(bpm: float, key_str: str, top_k=6) -> pd.DataFrame:
    mode = mode_from_key(key_str)
    def bpm_score(row):
        lo, hi = row["bpm_min"], row["bpm_max"]
        if lo <= bpm <= hi: return 1.0
        diff = min(abs(bpm - lo), abs(bpm - hi))
        return max(0.0, 1.0 - diff/30.0)
    def key_score(row):
        km = row["key_mode"]
        if km == "any": return 1.0
        return 1.0 if km == mode else 0.5
    rows = []
    for r in SEED_STYLES:
        bs = bpm_score(r); ks = key_score(r)
        score = 0.75*bs + 0.25*ks
        rr = r.copy(); rr["score"] = round(float(score), 3)
        rows.append(rr)
    df = pd.DataFrame(rows).sort_values("score", ascending=False).head(top_k)
    return df[["name","bpm_min","bpm_max","key_mode","score"]]

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Configuración (Balanced)")
    max_seconds = st.slider("Máx. duración a analizar (seg)", 10, 300, 90, 10)
    sr = st.selectbox("Sample rate (Hz)", [16000, 22050, 32000], index=1)
    bpm_min = st.number_input("Rango BPM min", value=60, min_value=30, max_value=200, step=1)
    bpm_max = st.number_input("Rango BPM max", value=180, min_value=60, max_value=300, step=1)
    show_plots = st.checkbox("Mostrar gráficos", value=True)

up = st.file_uploader("Sube tu audio (mp3, wav, m4a, ogg)", type=["mp3","wav","m4a","ogg"], accept_multiple_files=False)

if up:
    try:
        y, sr_eff = read_audio_any(up, target_sr=int(sr), mono=True)
    except Exception as e:
        st.error(f"No se pudo leer audio: {e}")
        st.stop()
    if max_seconds: y = y[: int(max_seconds * sr_eff)]
    y = normalize(y)

    c1, c2 = st.columns([1,1])
    with c1: st.audio(up)
    with c2: st.write({"sample_rate": sr_eff, "duración_s": round(len(y)/sr_eff,2), "frames": len(y)})

    n_fft, hop = 2048, 512
    freqs, times, mag = stft_mag(y, sr_eff, n_fft=n_fft, hop=hop)
    flux = spectral_flux(mag)
    bpm, beat_period = bpm_from_flux_acf(flux, sr_eff, hop, bpm_min=int(bpm_min), bpm_max=int(bpm_max))

    chroma_mat = chromagram(mag, freqs, fmin=50.0)
    key_str, key_conf = estimate_key_from_chroma(chroma_mat)

    m1, m2 = st.columns(2)
    with m1: st.metric("BPM (estimado)", f"{bpm:.1f}")
    with m2:
        st.metric("Tonalidad", key_str)
        st.caption(f"Confianza relativa: {key_conf:.3f}")

    st.subheader("🎯 Recomendaciones de estilos")
    recs = recommend_styles(bpm, key_str, top_k=6)
    if recs.empty:
        st.info("Sin recomendaciones para estos parámetros.")
    else:
        st.dataframe(recs, use_container_width=True)
    
    if show_plots:
        st.markdown("---"); st.subheader("📊 Visualizaciones")
        # Waveform + beat grid
        fig, ax = plt.subplots(figsize=(8,3))
        t_w = np.arange(len(y))/sr_eff
        ax.plot(t_w, y)
        if bpm > 0 and beat_period > 0:
            for b in np.arange(0, t_w[-1], beat_period): ax.axvline(b, linestyle='--', alpha=0.3)
        ax.set_xlabel("Tiempo (s)"); ax.set_ylabel("Amplitud")
        st.pyplot(fig, use_container_width=True); plt.close(fig)

        # Chromagram heatmap
        fig2, ax2 = plt.subplots(figsize=(8,3))
        im = ax2.imshow(chroma_mat, aspect='auto', origin='lower', interpolation='nearest')
        ax2.set_yticks(range(12)); ax2.set_yticklabels(PITCH_CLASSES)
        ax2.set_xlabel("Frames"); ax2.set_title("Chromagram")
        fig2.colorbar(im, ax=ax2)
        st.pyplot(fig2, use_container_width=True); plt.close(fig2)

st.markdown("---")
st.caption("Balanced: se enfocó en lo útil del día a día (BPM, tonalidad, recomendador). Si necesitas stems avanzados o exportar MIDI, podemos conectarlo como servicio externo.")