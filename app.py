import io
import math
import tempfile
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scipy import signal
import soundfile as sf

from pydub import AudioSegment

st.set_page_config(page_title="Analizador Musical (Lite)", page_icon="🎶", layout="wide")
st.title("🎶 Analizador de BPM, Tonalidad y Notas (Lite)")
st.caption("Edición liviana para Streamlit Cloud. Sin librosa/numba. MP3/M4A con ffmpeg (pydub).")

def read_audio_any(file, target_sr=22050, mono=True):
    name = file.name.lower()
    ext = name.split('.')[-1]
    try:
        seg = AudioSegment.from_file(file)
        if mono and seg.channels > 1:
            seg = seg.set_channels(1)
        if seg.frame_rate != target_sr:
            seg = seg.set_frame_rate(target_sr)
        samples = seg.get_array_of_samples()
        y = np.array(samples).astype(np.float32)
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
        sr = seg.frame_rate
        return y, sr
    except Exception as e:
        try:
            data, sr = sf.read(file, dtype="float32", always_2d=False)
            if mono and data.ndim > 1:
                data = data.mean(axis=1)
            if sr != target_sr:
                data = resample_audio(data, sr, target_sr)
                sr = target_sr
            return data.astype(np.float32), sr
        except Exception as e2:
            raise RuntimeError(f"No se pudo leer el audio. ffmpeg/sndfile necesarios. Detalle: {e} | {e2}")

def resample_audio(y: np.ndarray, sr_orig: int, sr_target: int) -> np.ndarray:
    if sr_orig == sr_target:
        return y
    g = math.gcd(sr_orig, sr_target)
    up = sr_target // g
    down = sr_orig // g
    return signal.resample_poly(y, up, down).astype(np.float32)

def normalize(y: np.ndarray) -> np.ndarray:
    m = np.max(np.abs(y)) if y.size else 0.0
    if m > 0:
        y = y / m
    return y.astype(np.float32)

def stft_mag(y: np.ndarray, sr: int, n_fft=2048, hop=512, win="hann"):
    f, t, Zxx = signal.stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft-hop, window=win, padded=False, boundary=None)
    mag = np.abs(Zxx)
    return f, t, mag

def spectral_flux(mag: np.ndarray):
    diff = np.diff(mag, axis=1)
    diff[diff < 0] = 0.0
    flux = diff.sum(axis=0)
    if flux.max() > 0:
        flux = flux / flux.max()
    return flux

def bpm_from_flux_acf(flux: np.ndarray, sr: int, hop: int, bpm_min=60, bpm_max=180):
    if flux.size < 4:
        return 0.0, np.array([])
    x = flux - flux.mean()
    acf = np.correlate(x, x, mode="full")
    acf = acf[acf.size//2:]
    lag_min = int(round((60.0/bpm_max) * sr / hop))
    lag_max = int(round((60.0/bpm_min) * sr / hop))
    lag_min = max(lag_min, 1)
    lag_max = min(lag_max, len(acf)-1)
    if lag_max <= lag_min:
        return 0.0, np.array([])
    lag = np.argmax(acf[lag_min:lag_max]) + lag_min
    bpm = 60.0 * sr / (lag * hop)
    if bpm < 75: bpm *= 2.0
    if bpm > 165: bpm /= 2.0
    period_s = 60.0 / max(bpm, 1e-6)
    return bpm, period_s

PITCH_CLASSES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
MAJOR_PROFILE = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
MINOR_PROFILE = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])

def chromagram(mag: np.ndarray, freqs: np.ndarray, fmin=50.0):
    chroma = np.zeros((12, mag.shape[1]), dtype=np.float32)
    valid = freqs >= fmin
    freqs_v = freqs[valid]
    mags_v = mag[valid, :]
    if freqs_v.size == 0:
        return chroma
    midi = 69 + 12*np.log2(freqs_v/440.0)
    pc = np.mod(np.rint(midi), 12).astype(int)
    for i in range(mags_v.shape[0]):
        chroma[pc[i], :] += mags_v[i, :]
    s = chroma.sum(axis=0, keepdims=True) + 1e-9
    chroma = chroma / s
    return chroma

def estimate_key_from_chroma(chroma_mat: np.ndarray):
    if chroma_mat.size == 0:
        return "Unknown", 0.0
    chroma_mean = np.mean(chroma_mat, axis=1)
    best_key = None
    best_score = -1e9
    scores = []
    for i, tonic in enumerate(PITCH_CLASSES):
        maj = np.corrcoef(chroma_mean, np.roll(MAJOR_PROFILE, i))[0,1]
        minr = np.corrcoef(chroma_mean, np.roll(MINOR_PROFILE, i))[0,1]
        scores.append((f"{tonic} major", maj))
        scores.append((f"{tonic} minor", minr))
        if maj > best_score:
            best_score = maj; best_key = f"{tonic} major"
        if minr > best_score:
            best_score = minr; best_key = f"{tonic} minor"
    scores.sort(key=lambda x: x[1], reverse=True)
    conf = float(max(0.0, scores[0][1] - scores[1][1]))
    return best_key, conf

def yin_f0(y: np.ndarray, sr: int, frame_size=2048, hop=256, thresh=0.1, fmin=80, fmax=1000):
    y = y.astype(np.float32)
    n = len(y)
    times = []
    f0s = []
    for start in range(0, n - frame_size, hop):
        frame = y[start:start+frame_size]
        tau_max = int(sr / max(1, fmin))
        tau_min = int(sr / max(1, fmax))
        tau_max = min(tau_max, frame_size-1)
        if tau_max <= tau_min+1:
            times.append(start/sr); f0s.append(np.nan); continue
        df = np.zeros(tau_max+1, dtype=np.float32)
        for tau in range(1, tau_max+1):
            diff = frame[:-tau] - frame[tau:]
            df[tau] = (diff*diff).sum()
        cmnd = np.zeros_like(df)
        cmnd[1:] = df[1:] * np.arange(1, len(df)) / (np.cumsum(df)[1:] + 1e-9)
        tau = None
        for t in range(max(tau_min,1), tau_max):
            if cmnd[t] < thresh:
                if 1 <= t < len(cmnd)-1:
                    a = cmnd[t-1]; b = cmnd[t]; c = cmnd[t+1]
                    denom = (a - 2*b + c)
                    if abs(denom) > 1e-12:
                        t = t - 0.5 * (c - a) / denom
                tau = t; break
        if tau is None:
            t = np.argmin(cmnd[max(tau_min,1):tau_max]) + max(tau_min,1)
            tau = float(t)
        f0 = sr / tau if tau and tau > 0 else np.nan
        times.append(start/sr)
        if f0 < fmin or f0 > fmax: f0 = np.nan
        f0s.append(f0)
    return np.array(times), np.array(f0s)

NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
def hz_to_note_name(f):
    if np.isnan(f) or f <= 0: return None
    m = 69 + 12*np.log2(f/440.0)
    midi = int(np.rint(m))
    name = NOTE_NAMES[midi % 12] + str(midi//12 - 1)
    return name

def segment_notes(times, f0s, min_dur=0.05):
    notes = []
    cur = None
    for t, f in zip(times, f0s):
        name = hz_to_note_name(f) if f and not np.isnan(f) else None
        if name is None:
            if cur:
                cur['end_s'] = t
                cur['duration_s'] = cur['end_s'] - cur['start_s']
                if cur['duration_s'] >= min_dur:
                    cur['freq_hz'] = float(np.nanmedian(cur.get('freqs', [np.nan])))
                    notes.append(cur)
                cur = None
            continue
        if cur is None:
            cur = {'note': name, 'start_s': t, 'freqs': [f] if not np.isnan(f) else []}
        elif name == cur['note']:
            if not np.isnan(f): cur['freqs'].append(f)
        else:
            cur['end_s'] = t
            cur['duration_s'] = cur['end_s'] - cur['start_s']
            if cur['duration_s'] >= min_dur:
                cur['freq_hz'] = float(np.nanmedian(cur.get('freqs', [np.nan])))
                notes.append(cur)
            cur = {'note': name, 'start_s': t, 'freqs': [f] if not np.isnan(f) else []}
    if cur:
        cur['end_s'] = times[-1] if times.size else 0.0
        cur['duration_s'] = cur['end_s'] - cur['start_s']
        cur['freq_hz'] = float(np.nanmedian(cur.get('freqs', [np.nan])))
        if cur['duration_s'] >= min_dur:
            notes.append(cur)
    if not notes:
        return pd.DataFrame(columns=['note','start_s','end_s','duration_s','freq_hz'])
    df = pd.DataFrame(notes)
    return df[['note','start_s','end_s','duration_s','freq_hz']]

with st.sidebar:
    st.header("⚙️ Configuración (Lite)")
    max_seconds = st.slider("Máx. duración a analizar (seg)", 10, 300, 90, 10)
    sr = st.selectbox("Sample rate (Hz)", [16000, 22050, 32000], index=1)
    show_plots = st.checkbox("Mostrar gráficos", value=True)
    extract_notes_flag = st.checkbox("Extraer notas (YIN-lite, monofónico)", value=False)
    note_fmin = st.number_input("Nota mínima (Hz)", value=80.0, min_value=20.0, max_value=2000.0, step=1.0)
    note_fmax = st.number_input("Nota máxima (Hz)", value=1000.0, min_value=50.0, max_value=5000.0, step=1.0)

up = st.file_uploader("Sube tu audio (mp3, wav, m4a, ogg)", type=["mp3","wav","m4a","ogg"], accept_multiple_files=False)

if up:
    try:
        y, sr_eff = read_audio_any(up, target_sr=int(sr), mono=True)
    except Exception as e:
        st.error(f"No se pudo leer audio: {e}")
        st.stop()
    if max_seconds:
        y = y[: int(max_seconds * sr_eff)]
    y = normalize(y)

    c1, c2 = st.columns([1,1])
    with c1: st.audio(up)
    with c2: st.write({"sample_rate": sr_eff, "duración_s": round(len(y)/sr_eff,2), "frames": len(y)})

    n_fft, hop = 2048, 512
    freqs, times, mag = stft_mag(y, sr_eff, n_fft=n_fft, hop=hop)
    flux = spectral_flux(mag)
    bpm, beat_period = bpm_from_flux_acf(flux, sr_eff, hop, bpm_min=60, bpm_max=180)

    chroma_mat = chromagram(mag, freqs, fmin=50.0)
    key_str, key_conf = estimate_key_from_chroma(chroma_mat)

    m1, m2 = st.columns(2)
    with m1: st.metric("BPM (estimado)", f"{bpm:.1f}")
    with m2:
        st.metric("Tonalidad", key_str)
        st.caption(f"Confianza relativa: {key_conf:.3f}")

    if show_plots:
        st.markdown("---"); st.subheader("📊 Visualizaciones")
        fig, ax = plt.subplots(figsize=(8,3))
        t_w = np.arange(len(y))/sr_eff
        ax.plot(t_w, y)
        if bpm > 0 and beat_period > 0:
            for b in np.arange(0, t_w[-1], beat_period): ax.axvline(b, linestyle='--', alpha=0.3)
        ax.set_xlabel("Tiempo (s)"); ax.set_ylabel("Amplitud")
        st.pyplot(fig, use_container_width=True); plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(8,3))
        im = ax2.imshow(chroma_mat, aspect='auto', origin='lower', interpolation='nearest')
        ax2.set_yticks(range(12)); ax2.set_yticklabels(PITCH_CLASSES)
        ax2.set_xlabel("Frames"); ax2.set_title("Chromagram")
        fig2.colorbar(im, ax=ax2)
        st.pyplot(fig2, use_container_width=True); plt.close(fig2)

    if extract_notes_flag:
        with st.spinner("Extrayendo notas (YIN-lite)…"):
            times_f0, f0s = yin_f0(y, sr_eff, frame_size=2048, hop=256, thresh=0.1, fmin=float(note_fmin), fmax=float(note_fmax))
            notes_df = segment_notes(times_f0, f0s, min_dur=0.05)
        if notes_df.empty:
            st.info("No se detectaron notas confiables. Usa pistas monofónicas (voz/bajo/lead).")
        else:
            st.subheader("🎼 Notas detectadas")
            st.dataframe(notes_df, use_container_width=True)
            csv = notes_df.to_csv(index=False).encode("utf-8")
            st.download_button("Descargar notas CSV", data=csv, file_name="notas_lite.csv", mime="text/csv")

st.markdown("---")
st.caption("Versión Lite — BPM, tonalidad y notas monofónicas (opcional).")

