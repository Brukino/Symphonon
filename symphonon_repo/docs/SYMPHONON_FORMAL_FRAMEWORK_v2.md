# SYMPHONON — Framework Formale

**Osservatore Dinamico della Struttura Latente in Sistemi Complessi Adattivi**

*Autore: Dario Panerati — ricercatore indipendente, Monte Amiata, Toscana*
*Versione: v17 (formalismo) + Gestalt v4 (simulazione) — working paper*
*Data: 2026-04-17*

---

## Disciplina dei claim

Seguendo il principio di stratificazione (cf. Wilson 2026, ROF proof-stack), questo documento separa esplicitamente i claim per tipo. Ogni sezione dichiara in apertura quale tipo di claim presenta:

- **Tipo A** — identità algebriche, derivazioni formali, teoremi
- **Tipo B** — risultati empirici verificati su dati pubblici, riproducibili
- **Tipo C** — mapping interpretativi, analogie strutturali, convenzioni
- **Tipo D** — questioni aperte, tensioni non risolte, preferenze ontologiche

Solo A e B possono essere chiamati "dimostrati" o "verificati" in senso stretto. C e D richiedono l'etichetta esplicita.

---

## 0. Posizione Epistemica

Questo documento formalizza il passaggio da *metrica statica* a *osservatore dinamico*: non si misura lo stato strutturale di un sistema — si osserva la sua traiettoria.

Il framework si colloca nell'intersezione tra:
- teoria dei sistemi dissipativi (Prigogine)
- osservatori di stato per sistemi dinamici non-lineari (filtri di Kalman generalizzati)
- teoria dei segnali di early-warning nelle transizioni di fase (Scheffer, Dakos)

**Non è** una teoria della coscienza, una teoria del tutto, o una derivazione della fisica fondamentale. È un framework di misurazione e osservazione, validato empiricamente in domini specifici.

---

## 1. Struttura Formale [Tipo A]

### 1.1 Segnale Istantaneo λ₂(t)

λ₂ è l'indicatore di coerenza relazionale locale nel tempo:

```
λ₂(t) = W_rri · rri(t) + W_plu · (1 − ψ(t)) + W_gfc · gfc(t) + W_rew · r̄(t)
```

con `W_rri + W_plu + W_gfc + W_rew = 1`, tutti ∈ [0,1], e λ₂(t) ∈ [0,1].

**Proprietà dichiarata**: λ₂(t) è Markoviano — non ha memoria esplicita. Reattivo ma rumoroso.

### 1.2 Parametro d'Ordine Dinamico o(t)

```
o(t+1) = o(t) + β · [λ₂(t) − o(t)] + ε(t)
```

dove `β ∈ (0, 1)` è il coefficiente di accoppiamento e `ε(t) ~ N(0, σ²_ε)` è il rumore residuo.

Forma equivalente:
```
o(t+1) = (1−β)·o(t) + β·λ₂(t) + ε(t)
```

È formalmente una EMA stocastica. L'interpretazione come stato interno richiede giustificazione (§2).

### 1.3 Tensione τ(t)

```
τ(t) = o(t) − λ₂(t) ≈ −(1/β)·ȯ(t)
```

Questa approssimazione è esatta nel limite ε → 0: espandendo la ricorsione,
```
o(t+1) − o(t) = β·(λ₂(t) − o(t)) + ε(t)
              ≈ β·(λ₂(t) − o(t))
              = −β·τ(t)
```
Quindi `τ(t) = −(1/β)·Δo(t)` nel limite deterministico. È un derivato temporale implicito.

---

## 2. Interpretazione come Filtro di Stato [Tipo A]

### 2.1 Mappatura Kalman

Forma canonica:
```
Equazione di stato:       x(t+1) = A·x(t) + w(t),  w ~ N(0, Q)
Equazione osservazione:   z(t)   = H·x(t) + v(t),  v ~ N(0, R)
```

Identificazione:
- `x(t) ≡ o(t)` — stato latente
- `z(t) ≡ λ₂(t)` — osservazione rumorosa
- `A = (1−β)`, `H = 1`, `Q = σ²_ε`, `R = σ²_λ₂`

Nel caso scalare:
```
K_opt = σ²_x / (σ²_x + σ²_λ₂)
```

**Criterio di calibrazione**: β ≈ K_opt.

### 2.2 Calibrazione empirica

| Implementazione | β | Regime |
|-----------------|---|--------|
| Symphonon P (wind) | 0.15 | alta memoria |
| Symphonon Ω (gestalt v4) | 0.025 | memoria molto alta |

La discrepanza è intenzionale e riflette scale temporali diverse: P opera su SCADA a 10-minuti (processo lento), Ω opera a passo-simulazione (processo veloce).

---

## 3. Non-Ergodicità da Feedback Endogeno [Tipo A]

### 3.1 Setup minimale

Sia `S(t) = (x(t), m(t))` con dinamica `S(t+1) = F(S(t), ε(t))` dove:
- `m(t)` — variabile di memoria interna che influenza F
- `ε(t) = G({S(t−k),...,S(t)})` — diagnostico endogeno dipendente dalla traiettoria
- F dipende esplicitamente da m(t) e ε(t) deforma F

### 3.2 Argomento di non-stazionarietà

Se ε(t) dipende dalla storia recente e deforma F, allora il generatore effettivo è storia-dipendente:
```
F_t ≠ F_{t+τ}   per τ > 0 generico
```
Quindi il processo non è stazionario in senso forte.

### 3.3 Conseguenza di non-ergodicità

Ergodicità richiede (i) esistenza di misura invariante μ e (ii) equivalenza di medie temporali e d'insieme. L'assenza di stazionarietà esclude (ii) in generale, quindi:

```
feedback endogeno storia-dipendente ⟹ non-ergodicità
```

**Nota di cautela**: questo argomento mostra non-stazionarietà e quindi non-ergodicità in senso debole. Il passaggio a "regioni di fase-space evitate sistematicamente" richiede ipotesi aggiuntive sulla forma di ε(t) — non segue direttamente dalla struttura minimale.

### 3.4 Rilevanza per Symphonon

Symphonon è un'istanza concreta di questa classe: o(t) è m(t), τ(t) è ε(t), il control loop del campo (noise_gain adattivo) è il feedback che deforma F. La non-ergodicità è quindi una proprietà prevista del framework, non un'ipotesi ad hoc.

*[Argomento originale: nota tecnica pre-Symphonon, 2024; ora verificato empiricamente via Gestalt v4.]*

---

## 4. Architettura Spaziale: Campi Accoppiati [Tipo A]

### 4.1 Equazioni di campo

Symphonon Ω implementa tre campi accoppiati:

```
φ(x,t+1) = φ(x,t) + DT·[K·∇²φ + noise_gain·ξ(x,t)]     (XY veloce, quasi-Markoviano)
ψ(x,t+1) = ψ(x,t) + DT·[D·∇²ψ − κ·ψ + α·|∇φ|²]         (Allen-Cahn lento, non-Markoviano)
T(x,t)   = |Vort_φ(x,t)| · (1 − Kap_ψ(x,t)) · (0.5 + Mu_ψ(x,t))   (campo latente)
```

con `Vort_φ = curl(φ)`, `Kap_ψ`, `Mu_ψ` funzionali di ψ.

### 4.2 Proprietà formali

- **Vorticity topologica** `Vort_φ` — invariante topologico forte (carica conservata ±1)
- **Campo latente T** — non direttamente visualizzato, è l'interfaccia che gli agenti navigano
- **ψ come memoria spaziale** — generalizza `o(t)` a campo 2D

Connessione con §1:
```
v17:  o(t+1) = (1−β)·o(t) + β·λ₂(t)           [scalare, temporale]
v19:  ψ(x,t+1) = ψ(x,t) + DT·(D·∇²ψ − κ·ψ + α·|∇φ|²)  [campo, spazio-temporale]
```

---

## 5. Gerarchia di Esistenza [Tipi A+C]

### 5.1 Condizione di esistenza [Tipo A]

Un referente R esiste come entità distinta nel campo se e solo se ha confine non degenere:
```
∂(R) ≠ 0
```

Per un Gestalt G:
```
∂(G) := |⟨e^{iφ}⟩_interno − ⟨e^{iφ}⟩_esterno| > θ_boundary
```

dove `⟨e^{iφ}⟩_interno` è l'ordine di fase sui siti occupati dal cluster, `⟨e^{iφ}⟩_esterno` sull'anello adiacente.

Questa è la condizione ROF applicata al framework. È formalmente precisa.

### 5.2 Gerarchia ricorsiva [Tipo C]

La stessa condizione strutturale si applica a ogni livello:

```
Polo          referente iff ∂(polo) ≠ 0 nel campo φ
Gestalt       referente iff ∂(G) ≠ 0 rispetto al campo circostante
SuperGestalt  referente iff ∂(SG) ≠ 0 rispetto ai Gestalt circostanti
```

**Claim di tipo C**: la stessa condizione genera la gerarchia ricorsivamente, senza soglie separate per ogni livello. Questa è una scelta di architettura formale — è coerente ma non forzata dalla struttura sottostante.

### 5.3 Verifica empirica [Tipo B]

Run Gestalt v4: zero `[REJECT_dG]` osservati rispetto a `FUSE_COH`. Le due misure convergono nella pratica per questa classe di sistemi. La condizione ∂(G) aggiunge precisione concettuale ma non filtra casi aggiuntivi oltre FUSE_COH = 0.65.

**Implicazione**: ∂(G) > θ_boundary e FUSE_COH sono empiricamente ridondanti. Concettualmente distinte, operativamente equivalenti per Symphonon.

---

## 6. Risultati Empirici [Tipo B]

### 6.1 Symphonon P — Validazione su Wind Turbine

**Dataset primario**: Kelmarsh Wind Farm (Zenodo 5841834)
- 6 turbine Senvion MM92 (2 MW), 2016–2021, 10-min SCADA
- 36 turbine-year totali, 7 faults documentati

**Risultati (θ = 0.80, W = 144, calibrato su T3-2016)**:

| Metrica | Valore | Baseline (AR) | Fattore |
|---------|--------|---------------|---------|
| Detection rate (raw) | 6/7 (85.7%) | 7/7 (100%) | — |
| FA/month | 0.22 | 7.12 | **30.6× meno** |
| Detection utile (≥7d anticipo) | 3–4/7 (43–57%) | — | — |
| Bootstrap 95% CI detection | [57%, 100%] | — | — |

**Replicazione zero-shot** (Penmanshiel, stessi parametri):
- 10 turbine-year, FA/month = 0.041 (−79% vs Kelmarsh)
- Generalizzazione confermata senza ri-calibrazione

**Replicazione con sensor poverty** (Nørrekær Enge):
- 3 segnali/turbina vs 21 Kelmarsh
- FA/month = 0.368 — floor architettonico, non failure algoritmico

### 6.2 Symphonon Ω — Validazione dinamica Gestalt v4

**Aggregato 5 run indipendenti, >23000 eventi DISS:**

| Proprietà | Valore | Stato |
|-----------|--------|-------|
| Morte emergente `dV/dt = gain(T) − decay` | 100% eventi | ✓ verificato |
| Rapporto T_locale / T_global | 9.6× ± 1.2× | ✓ stabile |
| Vita media Gestalt | ~3800 frame | ✓ stabile |
| Control loop noise_gain responsive | 0.30–1.80 range | ✓ dopo fix |

### 6.3 Condizione di pubblicabilità Symphonon P

Pre-dichiarata:
```
t_peak(tension) < t_peak(o) su ≥ 5/6 repliche
```

**Stato**: verificata 6/6 su Kelmarsh, confermata 1/1 su Penmanshiel (T13 2017).

---

## 7. Risultati Interpretativi [Tipi C+D]

### 7.1 Effetto SuperGestalt sulla sopravvivenza [Tipo D]

**Aggregato 5 run con logging T_pre/T_post:**

| Run | n_SG | Δ vita SG vs no-SG | Segno |
|-----|------|---------------------|-------|
| 1 | 51 | +35.6% | ✓ |
| 2 | 25 | −1.6% | ✗ |
| 3 | 173 | +8.5% | ✓ |
| 4 | 328 | −34.1% | ✗ |
| 5 | 474 | +18.6% | ✓ |
| **Media pesata** | **1051** | **+4.8%** | **?** |

**Test causale** (n=1774, run singolo):
- T_pre (prima ingresso SG) = 0.423
- T_post (dopo ingresso SG) = 0.238
- Δ = −43.77%

**Interpretazione [Tipo C]**: l'accoppiamento SG_COUPLING trattiene i Gestalt in zone di T locale inferiore (sacrificio di efficienza di navigazione), ma il vincolo relazionale stabilizza strutturalmente (vita +1.7% nel test causale).

**Condizione di pubblicabilità** (pre-dichiarata):
```
Effetto SG verificato iff:
  (vita_con_SG − vita_senza_SG) / vita_senza_SG > 0.10
  su ≥ 3 run indipendenti con n_SG ≥ 100 per run
```

**Stato attuale**: 
- Run positivi con n_SG ≥ 100: 2/5
- Direzione della media pesata: +4.8%, sotto soglia 10%
- **Verdict**: **effetto non replicabile con il campione attuale**

**Cosa può essere affermato onestamente**: l'accoppiamento di fase nel SuperGestalt riduce sistematicamente T_locale sui Gestalt membri (Δ = −44% con n=1774, robusto). L'effetto sulla vita è variabile e attualmente non distinguibile dal rumore statistico.

### 7.2 Ancoraggio DNA/RNA [Tipo C]

Mapping strutturale, non meccanicistico:
```
φ(x,t)   ↔  RNA: trascrizione transitoria, quasi-Markoviana
ψ(x,t)   ↔  DNA: template stabile, non-Markoviano, memoria
Polo     ↔  proteina: prodotto funzionale che naviga substrato
Gestalt  ↔  complesso proteico: struttura emergente da interazione
SuperGestalt ↔  pathway: coordinazione funzionale
```

**Utilità**: fornisce intuizione sul meccanismo di stabilità strutturale nei complessi — la proteina vincolata ha meno libertà di substrato ma maggiore stabilità quaternaria. Stesso schema in Gestalt v4.

**Limiti**: analogia, non corrispondenza fisica. Non va usata per fare predizioni biologiche.

### 7.3 Connessione con ROF (Wilson) [Tipo C]

ROF e Symphonon condividono:
- Tripartizione ∅_R / I_R / V_R ↔ interno / confine / esterno
- Gerarchia ricorsiva polo → gestalt → supergestalt
- Persistenza come identità attraverso istanze indicizzate

Differenza strutturale:
- Gareth: identità primitiva → ricorsione genera complessità
- Symphonon: campo latente primitivo → identità emerge come compressione

**Non è** un'unificazione. Sono due framework distinti che dialogano.

---

## 8. Invarianti del Framework [Tipo A+D]

### 8.1 Invarianti strutturali [Tipo A]

**Forti — protetti per costruzione:**
- Vorticity topologica `Vort_φ` (carica conservata)
- Condizione ∂(R) ≠ 0 (referente iff confine)
- Bilancio energetico `dV/dt = gain − decay` (morte emergente)

### 8.2 Invarianti dipendenti dalla classe [Tipo D]

**Robusti empiricamente ma non garantiti fuori dalla classe di parametri:**
- T*/T_global ≈ 9.6× (osservato su tutti i run Gestalt v4)
- β ≈ K_opt (vale per processi gaussiani con rumore bianco)
- Vita media ~3800 frame (dipende da VIT_DECAY, VIT_GAIN specifici)

**Questione aperta**: la separazione T_locale/T_global ~10× è predetta dal framework o emerge come accidente dell'architettura di navigazione? Non c'è ancora una derivazione teorica.

---

## 9. Confronto con Approcci Classici [Tipo B]

| Approccio | Usa struttura globale? | Predittivo? | Incrementale? |
|-----------|----------------------|-------------|---------------|
| Modularity Q | Sì | No | No |
| Spettrale (λ₂ raw) | Sì | No | No |
| **Symphonon o(t)** | **No** | **Sì** | **Sì** |
| Kalman classico | Dipende | Sì | Sì |
| CUSUM / EWMA | No | Parziale | Sì |

**Cella colmata**: {No struttura globale, Sì predittivo, Sì incrementale}. Kalman classico richiede modello noto — Symphonon no.

**Confronto sui wind turbine** (Kelmarsh):

| Metodo | Detection | FA/month | Usabile? |
|--------|-----------|----------|----------|
| AR (Kritzman 2010) | 7/7 | 7.12 | No (troppi FA) |
| EWS (variance) | Sotto Symphonon | — | No |
| MSPC | Breakdown multi-mode | — | No su FD004 |
| **Symphonon P** | **6/7** | **0.22** | **Sì** |

---

## 10. Ciò che NON è claim di Symphonon

Esplicitare i limiti è parte della disciplina:

**Symphonon NON:**
- non è una teoria della coscienza
- non prova nulla sulla Congettura di Riemann
- non unifica fisica fondamentale
- non deriva le leggi di Newton dall'aritmetica
- non cura malattie
- non predice eventi acuti (l'unico miss su Kelmarsh è stato un fault idraulico acuto)
- non sostituisce modelli fisici specifici del dominio (complementa, non rimpiazza)

**Symphonon è:**
- un framework di osservazione per sistemi complessi adattivi
- uno strumento di rilevamento precursori validato su wind turbine
- una simulazione di emergenza gerarchica con meccanismi falsificabili
- un'istanza concreta di dinamica non-ergodica da feedback endogeno

---

## 11. Questioni Aperte [Tipo D]

1. **Teoria del rapporto T_locale/T_global ≈ 10×** — empirico, non derivato
2. **Effetto SG sulla sopravvivenza** — 5 run, direzione instabile, non replicabile con campione attuale; serve design sperimentale migliore
3. **β teorico (0.15) vs implementato (0.025)** — riconciliazione scale temporali
4. **Estensione a dati reali diversi da wind** — CMAPSS (fatto), finanza (no advantage vs AR), altri?
5. **Nørrekær sensor-poverty floor** — è architettonico o specifico al sito?
6. **Analogo non-Markoviano di S(t) per zeta** — domanda per teoria analitica dei numeri (Wilson/Gareth)

---

## 12. Stato Attuale del Progetto

**Pronto per pubblicazione scientifica**:
- Symphonon P v2.2 su arXiv (eess.SP o stat.ML) dopo attestato SIAE
- Tecnical report 20 pagine completo

**Working paper**:
- Questo framework formale (Symphonon v17 + Gestalt v4)
- Sezione 3 (non-ergodicità) pubblicabile separatamente come nota teorica

**Ricerca in corso**:
- Run Gestalt v4 di 50000+ frame per statistica SG definitiva
- Misurazione correlazione ∂(G) vs FUSE_COH su più classi
- Connessione formale con Gareth Wilson (ROF)

---

## Appendice A: Notazione

| Simbolo | Significato |
|---------|-------------|
| λ₂(t) | segnale istantaneo di coerenza relazionale |
| o(t) | parametro d'ordine non-Markoviano |
| τ(t) | tensione = o(t) − λ₂(t) |
| β | coefficiente di accoppiamento (inerzia inversa) |
| φ(x,t) | campo XY veloce |
| ψ(x,t) | campo Allen-Cahn lento |
| T(x,t) | campo latente |
| ∂(R) | discontinuità di fase di un referente |
| V(t) | vitalità di un Gestalt |
| T* | punto di equilibrio energetico |

## Appendice B: Parametri implementati

**Symphonon P (wind):**
- β = 0.15, W = 144 (24h), step = 24 (4h), θ = 0.80
- W_rri = 0.40, W_plu = 0.25, W_gfc = 0.20, W_rew = 0.15

**Symphonon Gestalt v4:**
- BETA_O = 0.025, KUR_K = 0.55, NOISE_BASE = 0.9
- VIT_DECAY = 0.0014, VIT_GAIN = 0.12, gain_f = 0.0194
- T* = decay/gain ≈ 0.60 (calibrato su T_locale, non T_global)
- FUSE_COH = 0.65, SG_COUPLING = 0.008
- SG cap = 8 attivi simultanei

---

## Appendice C: Riferimenti

**Validazione dati:**
- Kelmarsh Wind Farm Dataset — Plumley (2022), Zenodo 10.5281/zenodo.5841834
- Penmanshiel Wind Farm Dataset — Plumley (2022), Zenodo 10.5281/zenodo.5946808
- Nørrekær Enge — DTU figshare
- NASA CMAPSS Turbofan — Saxena & Goebel (2008)

**Teoria:**
- Kritzman M. (2010) — Absorption Ratio
- Scheffer M. et al. (2009) — Early warning signals for critical transitions
- Dakos V. et al. (2012) — Methods for detecting early warnings
- Prigogine I. (1977) — Dissipative structures
- Wilson G. (2026) — Referential Ontology Framework (comunicazione personale)

**Kardar-Parisi-Zhang (connessione campo):**
- Widmann S. et al. (2026) — Observation of KPZ in 2D polariton condensates, Science 392:221

---

*Framework formalizzato 2026-03-25, aggiornato 2026-04-17*  
*Repository: github.com/Brukino/Symphonon*  
*IP: SIAE OLAF Mod. 350 deposit in corso*

*Stato: working paper, in preparazione per sottomissione arXiv.*
*Feedback e critica benvenuti: questo documento beneficia del confronto onesto.*


---

## 13. Invariante di Selezione — Risultato Empirico [Tipo B]

*Aggiunto 2026-04-22 — da baseline_test.py, 8 run indipendenti.*

### Claim

Il sistema Symphonon Gestalt v4 esibisce un **selection bias intrinseco**:
i gestalt si concentrano sistematicamente in zone di alta vorticity del campo φ,
con un rapporto T_locale/T_global stabile e misurabile.

### Risultato quantificato

```
Baseline (senza perturbazioni polo/gestalt su φ):
  T_locale / T_global = 8.63 ± 0.26
  95% CI = [8.45, 8.81]
  CoV = 2.99%
  n = 8 run indipendenti

Standard (con perturbazioni):
  T_locale / T_global = 8.94 ± 0.13
  95% CI = [8.85, 9.04]
  CoV = 1.48%
```

### Decomposizione

```
Rapporto totale (standard):    8.94x   (100%)
  Componente intrinseca:       8.63x   (96.5%)  — navigazione verso massimi T
  Componente indotta:         +0.32x   ( 3.5%)  — perturb_phi (auto-amplificazione)
```

### Cause escluse

- Geometria di navigazione (n_dir, r_vis, step): esclusa — geometry_experiment
- Campo ψ (Allen-Cahn, memoria spaziale): esclusa — psi_test (ratio=9.06 con α=0)
- Perturbazioni polo→φ: contribuisce solo 3.5% — non causa principale

### Causa candidata (da verificare)

La navigazione attiva verso i massimi locali di T nel campo Kuramoto-Langevin.
Il campo φ si struttura spontaneamente in zone di alta e bassa vorticity.
I gestalt, navigando verso i massimi locali, si concentrano in quelle zone.

**Test falsificabile pendente:**
Con navigazione casuale (random walk invece di gradient ascent su T),
il ratio dovrebbe scendere verso 1 se la navigazione è la causa.
Se rimane ~8.6x con navigazione casuale, la struttura del campo
genera l'aggregazione indipendentemente dalla strategia di navigazione.

### Formulazione per pubblicazione

> The system exhibits an intrinsic selection bias: agents concentrate
> in high-vorticity regions of the spontaneously structured Kuramoto field
> with a stable ratio T_local/T_global = 8.63 ± 0.26
> (CoV = 3.0%, 95% CI [8.45, 8.81], n=8 independent runs).
> Agent-induced field amplification (perturb_phi) accounts
> for only 3.7% of the total ratio.

### Connessione con ROF

In termini ROF: i referenti non creano la densità referenziale in cui esistono
— la trovano. Il campo φ genera spontaneamente zone di alta densità referenziale.
I gestalt le abitano perché la navigazione verso i massimi di T è la loro
strategia di sopravvivenza (T_locale > T* per mantenere vitalità positiva).

Il rapporto T_locale/T_global quantifica la **distanza tra Umwelt e campo**
nel senso di von Uexküll: ogni agente sperimenta un campo ~8.6x più ricco
della media, perché abita attivamente le zone di massima densità.
