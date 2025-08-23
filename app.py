import tempfile
# Análisis principal
# =============================
with st.spinner("Analizando tempo y tonalidad..."):
bpm, beat_times = estimate_bpm(y, sr_eff)
key_str, key_conf = estimate_key(y, sr_eff)


# Métricas
m1, m2 = st.columns(2)
with m1:
st.markdown('<div class="metric-card">', unsafe_allow_html=True)
st.metric("BPM (estimado)", f"{bpm:.1f}")
st.markdown('</div>', unsafe_allow_html=True)
with m2:
st.markdown('<div class="metric-card">', unsafe_allow_html=True)
st.metric("Tonalidad (clave)", key_str, help="Método Krumhansl-Schmuckler sobre cromas CQT")
st.caption(f"Confianza relativa: {key_conf:.3f} (Δ corr. top1-top2)")
st.markdown('</div>', unsafe_allow_html=True)


# Recomendaciones
st.subheader("🎯 Estilos recomendados")
recs = recommend_styles(bpm, key_str, styles_df)
if recs.empty:
st.info("No hay recomendaciones con la configuración actual.")
else:
for _, row in recs.iterrows():
st.markdown(
f"<span class='badge'>**{row['name']}** · {int(row['bpm_min'])}-{int(row['bpm_max'])} BPM · modo: {row['key_mode']} · score: {row['score']:.2f}</span>",
unsafe_allow_html=True,
)


# Visualizaciones
if show_plots:
st.markdown("---")
st.subheader("📊 Visualizaciones")
c1, c2 = st.columns(2)
with c1:
st.caption("Onda con beats")
fig, ax = plt.subplots(figsize=(8, 3))
librosa.display.waveshow(y, sr=sr_eff, ax=ax)
for bt in beat_times:
ax.axvline(bt, alpha=0.3, linestyle='--')
ax.set_xlabel("Tiempo (s)")
ax.set_ylabel("Amplitud")
st.pyplot(fig, use_container_width=True)


with c2:
st.caption("Cromagrama CQT (promedio usado para clave)")
chroma = librosa.feature.chroma_cqt(y=y, sr=sr_eff)
fig2, ax2 = plt.subplots(figsize=(8, 3))
img = librosa.display.specshow(chroma, x_axis='time', y_axis='chroma', cmap='magma', ax=ax2)
ax2.set_title('Chroma CQT')
fig2.colorbar(img, ax=ax2, format='%+0.2f')
st.pyplot(fig2, use_container_width=True)


# =============================
# Extracción de notas (opcional)
# =============================
if extract_notes_flag:
st.markdown("---")
st.subheader("🎼 Notas detectadas (monofónico)")
with st.spinner("Extrayendo notas con PYIN…"):
notes_df = extract_notes(y, sr_eff, threshold_prob=pyin_prob)
if notes_df.empty:
st.info("No se detectaron notas confiables. Si tu audio es polifónico (muchas fuentes) o tiene mezcla muy densa, el método puede fallar. Prueba con una pista monofónica o baja el umbral.")
else:
st.dataframe(notes_df, use_container_width=True)
# Descarga CSV
csv = notes_df.to_csv(index=False).encode('utf-8')
st.download_button("Descargar notas CSV", data=csv, file_name="notas_detectadas.csv", mime="text/csv")


if show_plots:
st.caption("Notas vs Tiempo (piano-roll simplificado)")
fig3, ax3 = plt.subplots(figsize=(10, 3))
for _, r in notes_df.iterrows():
ax3.hlines(r["midi"], r["start_s"], r["end_s"], linewidth=4)
ax3.set_xlabel("Tiempo (s)")
ax3.set_ylabel("MIDI note")
st.pyplot(fig3, use_container_width=True)


# Footer
st.markdown("---")
st.caption("Limitaciones: la detección de tonalidad en música polifónica puede ser ambigua; la extracción de notas con PYIN funciona mejor en señales MONOFÓNICAS (voz, bajo, lead sin acompañamiento). Para stems, considera separación previa (e.g., vocal/bajo) antes del análisis.")
