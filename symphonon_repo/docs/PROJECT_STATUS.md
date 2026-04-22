# SYMPHONON — Documento di Stato del Progetto
*Aggiornato: 2026-04-18*
*Uso: riferimento interno — condiviso con Gareth Wilson e Giulia*

---

## REGOLA D'USO

Questo documento è il riferimento unico per lo stato del progetto.
Prima di implementare qualcosa, verificare qui se è già stato fatto.
Prima di suggerire qualcosa, verificare qui se è già implementato.

---

## 1. SYMPHONON P — Predictive Maintenance

### Stato: validato, in preparazione per arXiv

**Risultati validati (non modificare senza nuovi dati):**

| Metrica | Valore | Dataset |
|---------|--------|---------|
| Detection rate | 6/7 (85.7%) | Kelmarsh 6T×6Y |
| Bootstrap 95% CI | [57%, 100%] | Kelmarsh |
| FA/month | 0.22 | Kelmarsh (θ=0.80) |
| FA reduction vs AR | 30.6× | Kelmarsh |
| Best advance (T3 2020) | +53.2 giorni | Kelmarsh |
| Penmanshiel (zero-shot) | 0.041 FA/month | Penmanshiel |
| CMAPSS FD004 useful | 71.1% | NASA CMAPSS |
| Nørrekær Enge FA | 0.368/month | DTU (3 segnali/turbina) |

**Condizione di pubblicabilità — verificata:**
```
t_peak(tension) < t_peak(o)  su ≥ 5/6 repliche → verificata 6/6
```

**File principali:**
- `symphonon_P_v4_1_local_runner.py` — runner principale Kelmarsh
- `symphonon_probe_v17.py` — probe core
- `wind_full_validation.py`, `wind_false_alarm.py`, `wind_ablation.py`
- `penmanshiel_validation.py`
- `symphonon_P_technical_report_v2_2.pdf` — report tecnico completo

**Prossimi passi:**
- [ ] Attestato SIAE OLAF → arXiv submission (eess.SP o stat.ML)
- [ ] Separare calibrazione/test (T3-2016 cal, 6 eventi validation)
- [ ] Definire "useful detection" ≥7d come metrica primaria

---

## 2. SYMPHONON Ω / GESTALT v4 — Simulazione di Campo

### Stato: attivo, diagnostica in corso

**File corrente:** `symphonon_gestalt_v4.py`
**Repository:** github.com/Brukino/Symphonon (commit 9b54361)

**Architettura dei campi:**
```
φ(x,t)  — campo XY veloce, quasi-Markoviano (Kuramoto-Langevin)
ψ(x,t)  — campo Allen-Cahn lento, non-Markoviano, memoria spaziale
T(x,t)  = |Vort_φ| · (1-Kap_ψ) · (0.5+Mu_ψ)  — campo latente, mai visualizzato
```

**Parametri correnti:**
```
BETA_O = 0.025, KUR_K = 0.55, NOISE_BASE = 0.9
VIT_DECAY = 0.0014, VIT_GAIN = 0.12, gain_f = 0.0194
T* = 0.601  (calibrato su T_locale, NON T_global)
FUSE_COH = 0.65, SG_COUPLING = 0.008
SG cap = 8 attivi simultanei
SG_MIN_FRAMES = 30, SG_MIN_D = 3.0
```

**Risultati empirici confermati (>15.000 eventi DISS):**

| Risultato | Valore | Stato |
|-----------|--------|-------|
| T_locale/T_global | ≈ 9.6× | ✓ stabile su tutti i run |
| Morte emergente (vitalita_zero) | 100% eventi | ✓ verificato |
| Vita media gestalt | ~3800 frame | ✓ stabile |
| Control loop noise_gain | 0.30-1.80 range | ✓ dopo fix |

**Effetto SG — NON ancora replicabile:**
```
Run 1: +35.6% (n_sg=51)   ✓
Run 2:  -1.6% (n_sg=25)   ✗
Run 3:  +8.5% (n_sg=173)  ✓
Run 4: -34.1% (n_sg=328)  ✗
Run 5: +18.6% (n_sg=474)  ✓
Media pesata: +4.8% — sotto soglia 10%, non pubblicabile
```

**Test causale T_pre/T_post (n=1774, robusto):**
```
T_pre  (prima SG) = 0.423
T_post (dopo SG)  = 0.238  (-43.77%)
→ SG trattiene in zone più povere ma stabilizza strutturalmente
```

**Log corrente** (formato aggiornato):
```
[FIELD] frame=N T=X T_gest=Y noise=Z poles=N gest=N sg=N rig_mean=A rig_max=B d_rig=C
[DISS]  frame=N vita=Nf T=X T_avg=Y T_pre=Z T_post=W rigidity=A d_rig=B sg=ID
[FUSE]  frame=N coh=X dG=Y n_poles=N gestalt_id=N
[SG]    Formato SuperGestalt N — coh=X dist=Y
```

**Metriche implementate** (dalla sessione corrente):
- `T_life_avg` — media T durante vita gestalt
- `T_pre_sg`, `T_post_sg` — T prima/dopo ingresso SG (test causalità)
- `rigidity` — 1 - corr(T_locale, T_global×9.6) nel tempo
- `d_rigidity` — derivata rigidity (rate of decoupling)
- `sg_entry_frame` — frame ingresso primo SG

**Prossimi passi:**
- [ ] Run 20000+ frame con diagnostica rigidity per prima analisi
- [ ] Eseguire `rigidity_analysis.py log.txt` sui nuovi log
- [ ] Verificare Check 1-4 di Gareth (pre-collapse signal, false positives, recovery, false(true))

---

## 3. METRICHE DI RIGIDITÀ — Nuova aggiunta 2026-04-17

### Stato: implementato, non ancora testato su dati reali

**Distinzione fondamentale** (non confondere):

| Tipo | Formula | Misura | Livello |
|------|---------|--------|---------|
| **Field rigidity** | `1 - corr(noise, power, W)` | Disaccoppiamento dal campo causale | Strutturale |
| **Fleet rigidity** | `1 - corr(P_t, P_flotta, W)` | Deviazione statistica dai pari | Anomalia index |

**La rigidità di flotta NON è collasso strutturale.**
Un guasto singolo su T3 produce alta fleet rigidity senza che la flotta collassi.
La rigidità di campo è la misura teoricamente corretta.

**File:**
- `rigidity_metrics.py` — implementazione corrente con rolling z-score
- `rigidity_analysis.py` — analisi Check 1-4 di Gareth sui log Gestalt v4
- `fleet_rigidity.py` — versione precedente (sostituita da rigidity_metrics.py)

**Normalizzazione corrente** (rolling, corretta):
```python
sensors_mean = sensors.rolling(W_SIGNAL, min_periods=W_SIGNAL//2).mean()
sensors_std  = sensors.rolling(W_SIGNAL, min_periods=W_SIGNAL//2).std()
sensors_z    = (sensors - sensors_mean) / (sensors_std + 1e-6)
```
NON usare std globale (data leakage — il futuro influenza il passato).

**Pending:** sostituire `+ 1e-6` con `sensors_std.clip(lower=0.01)` per sensori stabili.

---

## 4. FRAMEWORK FORMALE

### Stato: aggiornato, v2 con disciplina claim A/B/C/D

**File:** `SYMPHONON_FORMAL_FRAMEWORK_v2.md`

**Struttura claim (da usare sempre):**
- **Tipo A** — derivazioni formali, algebricamente dimostrabili
- **Tipo B** — risultati empirici verificati su dati pubblici
- **Tipo C** — mapping interpretativi, analogie strutturali
- **Tipo D** — questioni aperte, preferenze ontologiche

**Invarianti forti** (Tipo A):
- Vorticity topologica (carica conservata ±1)
- Condizione ∂(R) ≠ 0 (referente iff confine)
- Bilancio energetico `dV/dt = gain - decay`

**Invarianti empirici** (Tipo B, stabili ma non garantiti):
- T_locale/T_global ≈ 9.6× (tutti i run Gestalt v4)
- β ≈ K_opt (processi gaussiani con rumore bianco)

**Questioni aperte** (Tipo D):
- Perché T_locale/T_global ≈ 9.6×? Non derivato teoricamente.
- Effetto SG sulla sopravvivenza: instabile, non pubblicabile.
- β teorico (0.15) vs implementato (0.025): scale temporali diverse.

---

## 5. DIALOGO ROF-SYMPHONON (Wilson)

### Stato: attivo, nota congiunta in bozza

**File:** `ROF_Symphonon_joint_note.md`

**Mapping formalizzato:**
```
R (spazio referenziale)  ↔  configurazione campo (φ, ψ)
membrana/identità        ↔  ∂(G) — misurabile
volume/possibilità       ↔  distribuzione T_locale
r(t) relazione realizzata ↔  T_locale(t)
perdita fedeltà zoom-out ↔  T_locale/T_global ≈ 9.6× — misurata
mutabilità               ↔  T_locale relativo a T*
false(true)              ↔  rig_mean bassa, rig_max alta — misurabile
rigidità/disaccoppiamento ↔  1 - corr(T_locale, T_global)
```

**Domanda congiunta aperta:**
Perché T_locale/T_global ≈ 9.6×? ROF: teoria della selezione dello spazio referenziale. Symphonon: derivazione dall'equazione di navigazione. Nessuno dei due ha la risposta.

**Prossimi passi:**
- [ ] Gareth conferma mapping ROF nella nota congiunta
- [ ] Test OR formale: |T_A - T_B| < ε_OR come condizione di indeterminazione
- [ ] Predizioni P1-P3 verificabili in Gestalt v4 senza modifiche al codice

---

## 6. PANDEMIC EWS — Hackathon

### Stato: documento preparato, implementazione pending

**File:** `Symphonon_Pandemic_EWS_Hackathon.md`

**Traduzione componenti:**
```
noise_drift  → case trend anomaly (deviazione da baseline stagionale per regione)
compress(AR) → multi-region synchronisation (onset epidemico)
κ            → heterogeneity alarm (spread spaziale)
rigidity     → regional autonomy (disaccoppiamento dal campo nazionale)
```

**Rigidità regionale:**
```
rigidity_region(t) = 1 - rolling_corr(P_region(t), P_national(t), W)
```
Una regione che si disaccoppia dal campo nazionale prima che i casi cambino:
segnale strutturale di onset epidemico. Questo è il contributo nuovo.

**Dati disponibili (pubblici):**
- WHO FluNet (settimanale, 100+ paesi)
- ECDC COVID-19 (regionale, settimanale)
- US CDC ILINet (stato, settimanale)

**Prossimi passi:**
- [ ] Implementare P signal su ILINet (dati US)
- [ ] Validazione retrospettiva COVID-19 prima ondata (Jan-Mar 2020)
- [ ] Confronto: Symphonon vs z-score vs EWS vs AR su dati epidemiologici

---

## 7. COLLABORATORI

**Gareth Wilson (ROF):**
- Contributo: formalizzazione referenti, distinzione formale/naturale, false(true), collasso per over-coherence, Check 1-4 per rigidità
- Dialogo in corso su OR come soglia in campo rumoroso
- Lingua: inglese

**Giulia:**
- Contributo: lettura critica framework, connessione micelio/campo, analogia o(t) come accoppiamento progressivo
- Ha seguito il percorso dall'inizio con senso critico e disponibilità alla decostruzione
- Lingua: italiano

**Strumenti AI usati in parallelo:**
- Claude: implementazione, analisi log, documenti
- ChatGPT: suggerimento rolling z-score (recepito), d_rigidity già implementato quando suggerito
- Gemini: suggerimento z-score (valido ma con data leakage, corretto)

**Regola:** prima di implementare un suggerimento da strumenti AI, verificare questo documento.

---

## 8. FILE CHIAVE — MAPPA

```
symphonon_P_v4_1_local_runner.py  — runner principale Symphonon P
symphonon_probe_v17.py            — probe core
symphonon_gestalt_v4.py           — simulazione campo attuale
rigidity_metrics.py               — rigidità campo + flotta (v2, corretto)
rigidity_analysis.py              — analisi Check 1-4 Gareth su log Gestalt
SYMPHONON_FORMAL_FRAMEWORK_v2.md  — framework formale aggiornato
ROF_Symphonon_joint_note.md       — nota congiunta con Gareth
Symphonon_Pandemic_EWS_Hackathon.md — documento hackathon pandemic
```

---

## 9. COSA NON È ANCORA STATO TESTATO

Elenco esplicito per evitare di assumere che sia fatto:

- Rigidity_metrics.py su dati Kelmarsh reali (solo codice, no risultati)
- Rigidity_analysis.py su log Gestalt v4 con nuove metriche (log vecchi, senza rigidity)
- OR come condizione di campo in Gestalt v4
- Pandemic EWS su dati ILINet reali
- Effetto SG con n_SG ≥ 100 per run × 3 run positivi

---

*Aggiornare questo documento ad ogni sessione di lavoro.*
*Repository: github.com/Brukino/Symphonon*
