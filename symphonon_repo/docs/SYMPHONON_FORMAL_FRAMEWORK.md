# SYMPHONON — Framework Formale: Osservatore Dinamico della Struttura Latente
**v17 → formalizzazione teorica**  
*Prospettiva emergentista relazionale sui sistemi complessi*

---

## 0. Posizione Epistemica

Questo documento formalizza il passaggio da *metrica statica* a *osservatore dinamico*:
non si misura lo stato strutturale di un sistema — si **osserva la sua traiettoria**.

Il framework si colloca nell'intersezione tra:
- teoria dei sistemi dissipativi (Prigogine)
- osservatori di stato per sistemi dinamici non-lineari
- teoria dei segnali di early-warning nelle transizioni di fase (Scheffer, Dakos)

---

## 1. Definizioni Fondamentali

### 1.1 Segnale Istantaneo λ₂(t)

λ₂ è l'indicatore di coerenza relazionale **locale nel tempo**:

```
λ₂(t) = W_rri · rri(t)  +  W_plu · (1 − ψ(t))  +  W_gfc · gfc(t)  +  W_rew · r̄(t)
```

con `W_rri + W_plu + W_gfc + W_rew = 1`, tutti ∈ [0,1], e λ₂(t) ∈ [0,1].

**Proprietà**: λ₂(t) è Markoviano — non ha memoria. Ogni valore è indipendente
dal passato del sistema. Questo lo rende **reattivo ma rumoroso**.

### 1.2 Parametro d'Ordine Dinamico o(t)

L'equazione fondamentale del parametro d'ordine è:

```
o(t+1) = o(t) + β · [λ₂(t) − o(t)] + ε(t)
```

dove:
- `β ∈ (0, 1)` — coefficiente di accoppiamento (inerzia inversa)
- `ε(t) ~ N(0, σ²_ε)` — rumore stocastico residuo

**Forma equivalente** (riscrittura esplicita):

```
o(t+1) = (1−β) · o(t)  +  β · λ₂(t)  +  ε(t)
```

Questa è formalmente una **EMA stocastica**, ma con interpretazione fisica profonda:
o(t) non è una media — è uno **stato interno del sistema**.

---

## 2. Interpretazione come Filtro di Stato (Kalman-like)

### 2.1 Modello di Stato

Possiamo riscrivere il sistema nella forma canonica di un filtro di stato:

**Equazione di stato** (processo latente):
```
x(t+1) = A · x(t) + w(t),   w(t) ~ N(0, Q)
```

**Equazione di osservazione** (misura rumorosa):
```
z(t) = H · x(t) + v(t),     v(t) ~ N(0, R)
```

Nel caso Symphonon:
- `x(t) ≡ o(t)` — stato latente (struttura integrata)
- `z(t) ≡ λ₂(t)` — osservazione rumorosa
- `A = (1−β)` — transizione di stato (decadimento con memoria)
- `H = 1` — osservazione diretta (semplificata)
- `Q = σ²_ε` — rumore di processo
- `R = σ²_λ₂` — rumore di osservazione (stimato empiricamente)

**Update ottimale** (Kalman gain K):
```
K = P · Hᵀ · (H · P · Hᵀ + R)⁻¹
```

Nel caso scalare semplificato:
```
K_opt = σ²_x / (σ²_x + σ²_λ₂)
```

L'identificazione `β ≈ K_opt` diventa il **criterio di calibrazione** del sistema.

### 2.2 Interpretazione Fisica del Parametro β

| β piccolo (→ 0) | β grande (→ 1) |
|----------------|----------------|
| Alta inerzia   | Bassa inerzia  |
| Memoria lunga  | Memoria corta  |
| Filtra molto   | Segue λ₂       |
| Attrattore forte | Reattivo       |

**Valore implementato in v17**: β = 0.15 → regime di **alta memoria**, adeguato
per rilevare strutture persistenti su decine di episodi.

---

## 3. La Tensione come Derivata Latente

### 3.1 Definizione

```
tension(t) = o(t) − λ₂(t)
```

### 3.2 Derivazione analitica

Partendo dall'equazione del parametro d'ordine, al passo successivo:

```
o(t+1) − o(t) = β · [λ₂(t) − o(t)] = −β · tension(t)
```

Quindi:
```
tension(t) = −(1/β) · [o(t+1) − o(t)]  ≈  −(1/β) · ȯ(t)
```

**Risultato**: tension(t) è proporzionale alla **derivata temporale negata di o(t)**.

In forma continua (limite β → 0, dt → 0):
```
tension(t) ≈ −(τ / β) · do/dt
```

dove τ è la costante di tempo del sistema.

### 3.3 Interpretazione nei sistemi complessi

| Segno tension | Significato fisico | Predizione |
|--------------|-------------------|------------|
| tension > 0  | o > λ₂: inerzia strutturale — il sistema "ricorda" coerenza passata | Possibile caduta imminente |
| tension ≈ 0  | Equilibrio locale: o ≈ λ₂ | Regime stabile |
| tension < 0  | o < λ₂: λ₂ salito, o non ha ancora risposto | Rafforzamento in corso |

**Spike di |tension|** → il sistema è in **transizione strutturale**.

---

## 4. Early Warning Signals: Teoria

### 4.1 Framework teorico (Dakos et al., 2012)

Nei sistemi complessi prossimi a biforcazioni, si osserva **Critical Slowing Down (CSD)**:
il sistema impiega sempre più tempo a tornare all'equilibrio dopo perturbazioni.

Indicatori classici:
- Aumento della varianza: `Var[x(t)] ↑`
- Aumento dell'autocorrelazione: `AC1[x(t)] ↑`
- Aumento del lag-1 return time

### 4.2 Symphonon come CSD detector agnostico

Il trio (λ₂, o, tension) implementa un rilevatore di CSD **senza conoscere la struttura**:

```
CSD ↔ tension(t) persiste a lungo senza annullarsi
    ↔ std(tension) aumenta nelle finestre temporali
    ↔ |tension| anticipa GT transition
```

**Proprietà chiave** (verificata empiricamente su v17):

```
std(o) < std(λ₂)  AND  ρ(o, GT) > ρ(λ₂, GT)
```

Questo è impossibile per semplice smoothing passivo — implica che o(t) **estrae
informazione strutturale**, non solo comprime rumore.

---

## 5. Proprietà Formali del Sistema

### 5.1 Convergenza all'attrattore

Se λ₂(t) = λ̄ (costante), allora o(t) converge geometricamente:

```
o(t) → λ̄ + O((1−β)ᵗ)
```

Tempo caratteristico di convergenza:
```
τ_conv = −1 / ln(1−β) ≈ 1/β  per β piccolo
```

Per β = 0.15: τ_conv ≈ 6.2 passi → **memoria efficace ~6 episodi**.

### 5.2 Stabilità di o: l'indicatore o_stability

```
o_stability = 1 − std_{window}(o)
```

Interpetazione:
- `o_stability > 0.80` → attrattore raggiunto (struttura cristallizzata)
- `o_stability < 0.50` → sistema in transizione (zona caotica/critica)

### 5.3 Teorema informale (da dimostrare formalmente)

**Proposizione**: Sotto le ipotesi di stazionarietà locale e rumore additivo gaussiano,
il parametro d'ordine o(t) è lo **stimatore a minima varianza** della struttura latente
del sistema, condizionato alla traiettoria osservata λ₂(0:t).

*Dimostrazione sketch*: Segue dalla teoria del filtro di Kalman in regime stazionario,
identificando σ²_λ₂ come varianza del rumore di misura e σ²_struct come varianza
del processo strutturale sottostante.

---

## 6. Il Trio come Framework Unificato

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   λ₂(t)  ──→  segnale istantaneo (Markoviano, rumoroso) │
│                        │                               │
│                        ▼  β-coupling                   │
│   o(t)   ──→  stato latente (non-Markoviano, filtrato)  │
│                        │                               │
│                        ▼  differenza                   │
│ tension(t) ──→  gradiente temporale implicito           │
│               ≈ −(1/β) · ȯ(t)                          │
│               = early warning di transizione            │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Proprietà emergenti del trio**:

1. **Separazione scale temporali**: λ₂ coglie dinamica veloce, o quella lenta
2. **Robustezza al rumore**: std(o) < std(λ₂) per costruzione
3. **Predittività**: tension anticipa GT transition senza accesso alla struttura globale
4. **Domain-agnosticism**: funziona su LFR, Zachary, e (ipotesi) su reti reali

---

## 6b. Condizione di Esistenza del Gestalt (integrazione ROF)

*Aggiunta 2026-04-15 — in dialogo con il Referential Ontology Framework (ROF) di G. Wilson.*

### Motivazione

La soglia di fusione FUSE_COH = 0.65 usata nell'implementazione è una soglia empirica calibrata sulla coerenza interna dei poli. Misura quanto i poli siano in fase tra loro, ma non formalizza quando il cluster diventa un referente distinto rispetto al campo che lo circonda.

Il ROF fornisce una condizione più precisa: un referente esiste se e solo se il suo confine è non degenere — `∂(G) ≠ 0` (ROF7). Applicato al Gestalt, questo significa che il cluster esiste come entità distinta quando produce una discontinuità identificabile nel campo φ, non solo quando i suoi membri sono internamente coerenti.

### Condizione formale

```
Un cluster di poli forma un Gestalt G se e solo se:

∂(G) ≠ 0

dove ∂(G) è la discontinuità di fase tra campo
interno al cluster e campo φ circostante:

∂(G) := |⟨e^{iφ}⟩_interno − ⟨e^{iφ}⟩_esterno|

⟨e^{iφ}⟩_interno = ordine di fase medio sui siti
  occupati dai poli del cluster

⟨e^{iφ}⟩_esterno = ordine di fase medio sull'annello
  di celle adiacenti al cluster (non occupate)

G è un referente distinto iff ∂(G) > θ_boundary
```

### Interpretazione nella tripartizione ROF

```
∅_R  (interno)  : vitalità e genotipo collettivo — il "punto" del Gestalt
I_R  (confine)  : discontinuità di fase ∂(G) — ciò che rende il Gestalt
                  distinguibile dal campo circostante
V_R  (esterno)  : campo φ circostante — spazio di possibilità in cui naviga
```

Il Gestalt esiste come condizione di possibilità attiva quando ha un confine misurabile. Senza confine — coerenza interna alta ma indistinguibile dal campo — il cluster non è un referente distinto: è una fluttuazione del campo.

### Ricorsività gerarchica

La condizione si applica a ogni livello:

```
Polo         → referente iff ∂(polo) ≠ 0 nel campo φ
Gestalt      → referente iff ∂(G) ≠ 0 rispetto al campo circostante
SuperGestalt → referente iff ∂(SG) ≠ 0 rispetto ai Gestalt circostanti
```

La stessa condizione strutturale genera la gerarchia — non soglie separate calibrate empiricamente per ogni livello.

### Nota metodologica

La correlazione empirica tra FUSE_COH e ∂(G) deve essere verificata prima di modificare l'implementazione. Le due misure potrebbero convergere nella pratica pur rimanendo distinte concettualmente. Verifica computazionale in corso.

---

## 6c. Dinamica di Persistenza del Gestalt — Risultati Empirici

*Aggiunta 2026-04-16 — da validazione su Symphonon Gestalt v4.*

### Motivazione

L'analisi empirica della simulazione ha prodotto risultati parzialmente inattesi che richiedono documentazione rigorosa, incluse le ipotesi non confermate.

### Risultato 1 — Bilancio energetico continuo (morte emergente) ✓ CONFERMATO

La dissoluzione del Gestalt non richiede soglie arbitrarie. Il meccanismo corretto è un bilancio energetico continuo:

```
dV/dt = gain(T_locale) − decay

gain(T) = α · T_locale       α = VIT_GAIN · gain_f
decay   = VIT_DECAY · decay_f

Punto di equilibrio: T* = decay / α
```

**Nota critica di calibrazione**: T_locale sui Gestalt è sistematicamente ~9.6× superiore a T_global, perché i Gestalt navigano attivamente verso i massimi locali di vorticity. La calibrazione di T* deve usare T_locale osservato, non T_global. Confondere le due scale produce parametri fuori range di un ordine di grandezza.

Parametri calibrati empiricamente:
```
T_gest_medio ≈ 0.69    T_gest_min ≈ 0.55
T_global_medio ≈ 0.088  Rapporto ≈ 9.6×

gain_f = 0.0194  →  T* = 0.601
```

**Distribuzione vita osservata** (aggregato 4 run, n=5225 eventi):
```
Vita media:   ~3600 frame
Vita mediana: ~3500 frame
Distribuzione: ~98% tra 1000-5000 frame
```

Tutti gli eventi di dissoluzione hanno `T_locale ≈ 0` al momento della morte — il campo locale si svuota completamente prima della dissoluzione. Il meccanismo è verificato e stabile su tutti i run.

### Risultato 2 — Effetto SG sulla sopravvivenza ✗ NON REPLICABILE

**Ipotesi iniziale**: l'accoppiamento di fase SG_COUPLING=0.008 riduce la varianza nella navigazione, aumentando T_locale medio e quindi la vita dei Gestalt membri.

**Dati aggregati (4 run, n_SG_totale=577)**:

```
Run 1: +35.6%  (n_sg=51)   ✓
Run 2:  -1.6%  (n_sg=25)   ✗
Run 3:  +8.5%  (n_sg=173)  ✓
Run 4: -34.1%  (n_sg=328)  ✗

Media pesata: -13.8%
Run positivi con n_sg≥100: 1/4
```

**Conclusione**: l'effetto SG sulla sopravvivenza non è replicabile. La direzione è instabile tra run. L'ipotesi meccanicistica non trova supporto nei dati aggregati. Il risultato del run 1 (+35.6%) era un artefatto del campione piccolo.

**Questione aperta**: perché l'effetto cambia segno? Una possibilità è che i Gestalt entrino in SG preferenzialmente nelle fasi dinamiche del campo (alta T, alta mobilità) — il che significherebbe che la classificazione SG è correlata con la posizione nel ciclo del campo, non con un effetto causale dell'accoppiamento. Richiede un test di causalità esplicito.

### Risultato 3 — Separazione scale spaziali T_locale vs T_global ✓ CONFERMATO

```
T_global(t)  →  densità media di vorticity nel campo (scala globale)
T_locale(g)  →  qualità della posizione del Gestalt (scala locale)
Rapporto:    ≈ 9.6× stabile su tutti i run
```

Questi sono osservabili distinti. Il control loop del campo deve essere calibrato su T_global; la dinamica di vita dei Gestalt deve essere calibrata su T_locale.

### Ancoraggio fenomenologico — DNA/RNA

Per interpretare la gerarchia polo→gestalt→supergestalt in termini di sistemi biologici osservabili:

```
φ(x,t)  — campo veloce, quasi-Markoviano
        ↔ RNA: trascrizione attiva, espressione transitoria

ψ(x,t)  — campo lento, non-Markoviano, memoria spaziale
        ↔ DNA: template stabile, portatore di storia

Polo    — condizione di possibilità che naviga T
        ↔ proteina: prodotto dell'espressione, agente funzionale

Gestalt — organismo da fusione di poli, identità collettiva
        ↔ complesso proteico: struttura emergente da interazione

SuperGestalt — relazione tra Gestalt coerenti
        ↔ pathway: coordinazione funzionale tra complessi
```

**Implicazione per il test dell'effetto SG**: nel sistema DNA-RNA, l'appartenenza a un pathway non aumenta necessariamente la stabilità delle singole proteine — può aumentare la loro *efficacia funzionale* pur riducendone la persistenza (turnover accelerato per maggiore attività). L'effetto SG potrebbe non manifestarsi nella vita media ma nella *qualità della navigazione* — T_locale medio durante la vita, non durata. Questo suggerisce una metrica di test alternativa da verificare.

### Prossimi passi

1. Misurare T_locale medio *durante* la vita dei Gestalt (non solo al momento della morte), separando con SG vs senza SG
2. Test di causalità: i Gestalt entrano in SG quando T è già alto, o T sale dopo l'ingresso in SG?
3. Verificare se SG_COUPLING troppo basso (0.008) per produrre effetto misurabile sulla navigazione

---

## 6c. Dinamica di Persistenza del Gestalt — Risultati Empirici

*Aggiunta 2026-04-16 — da validazione su Symphonon Gestalt v4.*

### Motivazione

L'analisi empirica della simulazione ha rivelato tre risultati non previsti dal design che meritano formalizzazione nel framework.

### Risultato 1 — Bilancio energetico continuo (morte emergente)

La dissoluzione del Gestalt non richiede soglie arbitrarie. Il meccanismo corretto è un bilancio energetico continuo:

```
dV/dt = gain(T_locale) − decay

gain(T) = α · T_locale       α = VIT_GAIN · gain_f
decay   = VIT_DECAY · decay_f

Punto di equilibrio: T* = decay / α
```

Il Gestalt persiste se `T_locale > T*`, si dissolve se `T_locale < T*` per tempo sufficiente a portare V → 0.

**Nota critica di calibrazione**: T_locale sui Gestalt è sistematicamente ~9.6× superiore a T_global, perché i Gestalt navigano attivamente verso i massimi locali di vorticity. La calibrazione di T* deve usare T_locale osservato, non T_global.

Parametri calibrati empiricamente (Symphonon Gestalt v4, run su 6100 frame):
```
T_gest_medio = 0.725    T_gest_min = 0.551    T_gest_max = 0.972
T_global_medio = 0.088  Rapporto T_gest/T_global ≈ 9.6×

gain_f = 0.0194  →  T* = 0.601
(tra T_gest_min e T_gest_medio — mortalità nelle fasi povere del campo)
```

**Distribuzione vita osservata** (n=558 eventi):
```
Vita media:   4050 frame
Vita mediana: 4127 frame
Range:        694 — 6126 frame
< 1000f:       2%
1000—5000f:   56%
> 5000f:      42%
```

Tutti i 558 eventi di dissoluzione hanno `T_locale < T*` al momento della morte — il meccanismo è verificato.

### Risultato 2 — Effetto protettivo del SuperGestalt

L'accoppiamento di fase nel SuperGestalt produce un vantaggio di sopravvivenza misurabile sui Gestalt membri, **non programmato esplicitamente**:

```
Vita media Gestalt in SuperGestalt:   5319 frame  (n=51)
Vita media Gestalt senza SuperGestalt: 3922 frame  (n=507)

Differenza: +1397 frame  (+35.6%)
```

**Meccanismo ipotizzato** (non ancora verificato formalmente): l'accoppiamento di fase SG_COUPLING=0.008 riduce la varianza nella navigazione del campo. I Gestalt accoppiati si muovono in modo più coerente verso zone di T alto, mantenendo T_locale sistematicamente più alto rispetto a Gestalt isolati.

**Significato per il framework**: la coerenza relazionale di secondo ordine (SuperGestalt) aumenta la persistenza strutturale dei componenti di primo ordine (Gestalt) senza che questo sia un effetto codificato. È un esempio di emergenza top-down: la struttura relazionale superiore influenza la dinamica dei livelli inferiori.

**Stato**: risultato replicato su due run indipendenti con direzione consistente. Campione SG ancora limitato (n=51). Richiede run più lungo per robustezza statistica.

### Risultato 3 — Separazione scale spaziali T_locale vs T_global

La scoperta che T_locale_gestalt ≈ 9.6 × T_global ha implicazioni per il framework generale:

```
T_global(t)  →  misura della densità media di vorticity nel campo
T_locale(g)  →  misura della qualità della posizione del Gestalt nel campo
```

Questi sono osservabili distinti con scale diverse. Il control loop del campo deve essere calibrato su T_global; la dinamica di vita dei Gestalt deve essere calibrata su T_locale. Confonderli produce parametri fuori scala di uno o due ordini di grandezza.

### Condizione di pubblicabilità

```
Effetto SG verificato iff:
  (vita_con_SG − vita_senza_SG) / vita_senza_SG > 0.10
  su ≥ 3 run indipendenti con n_SG ≥ 100 per run
```

*Stato attuale: 2/3 run con direzione positiva, n_SG insufficiente.*

---

## 7. Confronto con Approcci Classici

| Approccio | Usa struttura globale? | Predittivo? | Incrementale? |
|-----------|----------------------|-------------|---------------|
| Modularity Q | Sì (comunità esplicite) | No | No |
| Spettrale (λ₂ raw) | Sì (Laplaciano) | No | No |
| **Symphonon o(t)** | **No** | **Sì** | **Sì** |
| Kalman filter classico | Dipende | Sì | Sì |
| CUSUM / EWMA | No | Parziale | Sì |

**Gap colmato**: Symphonon opera nella cella {No struttura globale, Sì predittivo, Sì incrementale} — precedentemente vuota.

---

## 8. Esperimento Cruciale: Scenario Shock

Per falsificare/validare il framework, il test definitivo è:

**Protocollo**:
1. Regime stabile per T₁ passi → `tension ≈ 0`, `o_stability > 0.80`
2. Shock improvviso al passo t*: jump/drop di λ₂ di ampiezza Δ
3. Ritorno graduale per T₂ passi

**Predizioni verificabili**:
```
t_response(λ₂)   ≈ 1 passo           (reattività immediata)
t_response(o)    ≈ τ_conv ≈ 1/β passi (inerzia strutturale)
t_response(tension): picco a t* → anticipa la transizione di o
```

**Condizione di pubblicabilità**:
```
t_peak(tension) < t_peak(o)   su ≥ 5/6 repliche
```

---

## 9. Roadmap verso il Framework Completo

### Step 1 (implementato): EMA stocastica
```python
o[t+1] = o[t] + beta * (lambda2[t] - o[t]) + epsilon
```

### Step 2 (prossimo): Kalman adattivo
Stimare β ottimale online da varianza empirica di λ₂:
```python
beta_opt = var_struct / (var_struct + var_lambda2)
```

### Step 3 (futuro): Osservatore non-lineare
Estendere a sistemi con dynamics non-stazionarie:
```
o(t+1) = f(o(t), λ₂(t)) + w(t)
```
con f appresa da dati (LSTM / SSM).

### Step 4 (vision): Multi-scala
Trio gerarchico: (λ₂_fast, o_mid, Ω_slow) come campo φ + campo ψ di Symphonon Ω v19.

---

## 10. Connessione con Symphonon Ω v19

Il campo lento ψ in v19 è la **generalizzazione spaziale** di o(t):

```
v17:  o(t+1)   = (1−β)·o(t)   + β·λ₂(t)           [scalare, temporale]
v19:  ψ(x,t+1) = ψ(x,t) + DT·(D·∇²ψ − κ·ψ + α·φ)  [campo, spazio-temporale]
```

La tensione scalare diventa un **campo di tensione**:
```
Tension_field(x,t) = ψ(x,t) − φ(x,t)
```

Dove i picchi del campo di tensione indicano **nucleazione di nuove strutture semantiche**.

---

## Appendice: Notazione e Parametri v17

| Simbolo | Valore | Significato |
|---------|--------|-------------|
| β | 0.15 | Coefficiente accoppiamento o(t) |
| σ_ε | 0.02 | Rumore stocastico residuo |
| θ_crystal | 0.65 | Soglia stato cristallizzato |
| θ_critical | 0.40 | Soglia zona critica |
| W_rri | 0.40 | Peso Relational Resonance Index |
| W_plu | 0.25 | Peso pluralismo (1−ψ) |
| W_gfc | 0.20 | Peso Global Field Coherence |
| W_rew | 0.15 | Peso reward normalizzato |

---

*Framework formalizzato da analisi sperimentale Symphonon v17*  
*Data: 2026-03-25*  
*Stato: Working paper — in preparazione per validazione su dati reali*
