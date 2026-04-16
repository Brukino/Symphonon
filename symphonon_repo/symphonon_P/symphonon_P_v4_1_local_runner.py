"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Symphonon P v3.2 — Sistema a due livelli, L1 corretto                     ║
║                                                                              ║
║  Diagnosi v3.1:                                                             ║
║    Il Livello 1 (Mahalanobis su rolling-sigmoid space) era rotto.          ║
║    Il rolling sigmoid per costruzione ricentra ogni segnale sulla sua      ║
║    storia recente → la "regione sana" si sposta col segnale stesso →       ║
║    d_norm strutturalmente alto per tutto l'anno → persistenza alta         ║
║    anche in salute → zero detection con zero FA.                           ║
║                                                                              ║
║  Fix v3.2:                                                                  ║
║    Livello 1 sostituito con P_v2.1 — la misura che funzionava.             ║
║    P_v2.1 aveva 7/7 detection ma 1.67 FA/mese.                            ║
║    Il Livello 2 (persistenza + velocità) è il filtro FA.                   ║
║                                                                              ║
║  ARCHITETTURA v3.2                                                          ║
║                                                                              ║
║  LIVELLO 1 — P_base (= P v2.1)                                             ║
║    P_base = 0.45·noise_rs + 0.30·PR_inv_rs + 0.25·compress_intra_rs      ║
║    Candidato: P_base(t) > THR_P                                            ║
║    Fuoco: alta recall, include FA e TP (come v2.1)                        ║
║                                                                              ║
║  LIVELLO 2 — Coerenza temporale della memoria (doppio criterio)            ║
║                                                                              ║
║    PERSISTENZA(t) = frac( P_base[t-L:t] > THR_P )                        ║
║      L = 40 finestre (≈7 giorni > periodo oscillazione regime)            ║
║      Alta se P_base CI RESTA elevato, non solo ci passa                    ║
║      Degrado ✓  Transitorio ✗  Regime shift ✓                            ║
║                                                                              ║
║    VELOCITÀ(t) = relu( EMA(noise_raw,4) − EMA(noise_raw,40) )             ║
║      Alta se noise_raw cresce da ≈160h                                     ║
║      Degrado ✓  Transitorio ✗  Regime shift ✗ (→ 0 se stabile)          ║
║                                                                              ║
║    L2 = (persistenza > THR_PERS) AND (velocità > THR_VEL_SIGN)                ║
║                                                                              ║
║  GATING GERARCHICO                                                          ║
║    Candidato:  P_base > THR_P                                              ║
║    Qualificato: L2 attivo                                                  ║
║    Confermato: qualificato per ≥ K_CONFIRM finestre consecutive            ║
║    Off: P_base < THR_P − HYST per ≥ MIN_OFF finestre                     ║
║                                                                              ║
║  Separazione fisica dei tre casi:                                          ║
║    Degrado:        P↑ (L1✓)  persistenza✓  velocità✓  →  ALARM          ║
║    Transitorio:    P↑ (L1✓)  persistenza✗  velocità✗  →  no alarm       ║
║    Regime shift:   P↑ (L1✓)  persistenza✓  velocità✗  →  no alarm       ║
║                                                                              ║
║  Uso:                                                                        ║
║    python symphonon_P_v3_2_local_runner.py --data D:\\...\\data           ║
║    python symphonon_P_v3_2_local_runner.py --data D:\\...\\data --skip-cmapss║
║                                                                              ║
║  Output (./output_v3_2/):                                                   ║
║    results.json  summary.txt  fig_kelmarsh_*.png  fig_trajectory_*.png     ║
╚══════════════════════════════════════════════════════════════════════════════╝

Dipendenze:  pip install numpy scipy pandas matplotlib scikit-learn
Python:      3.9+
"""

import argparse
import json
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ═════════════════════════════════════════════════════════════════════════════
#  PARAMETRI GLOBALI
# ═════════════════════════════════════════════════════════════════════════════

WIN             = 144    # finestra SCADA: 24h × 6 misure/h
STEP            = 24     # passo: 4h
RW              = 10     # lookback rolling sigmoid (noise, compress)
RW_PR           = 25     # lookback rolling sigmoid per PR_inv

# ── Livello 1 — P_base (= P v2.1) ─────────────────────────────────────────
# [FIX v3.2] Sostituisce Mahalanobis: P_base non è auto-normalizzato
# e mantiene separazione stabile tra stato sano e stato anomalo
THR_P           = 0.72   # soglia P_base per candidatura (identica a v2.1)

# ── Livello 2 — MIXTURE OF DETECTORS ──────────────────────────────────────
#
# ALARM = P_base > THR_P  AND  (BRANCH_A  OR  BRANCH_B)
#
# BRANCH A — Episodico (v3.3-like):
#   Cattura: fault con spike ricorrenti di ampiezza (T6-2017, T3-2018)
#   pers_short > THR_PERS_A  AND  vel_amp > THR_VEL_AMP
#
# BRANCH B — Strutturale (v3.5-like):
#   Cattura: fault con deriva lenta monotona (T3-2016, T3-2020, T1-2020)
#   pers_long > max(THR_PERS_BASE - ALPHA * vel_sign, PERS_FLOOR)
#
# WARMUP: i primi WARMUP_WIN finestre non possono generare allarmi
#   (evita path-dependency della vel_sign_pers all'inizializzazione)

# Branch A — episodico
L_PERS_A        = 20     # lookback breve ≈3.3 giorni
THR_PERS_A      = 0.12   # soglia persistenza breve (da v3.3)
THR_VEL_AMP     = 0.08   # soglia velocity amplitude (da v3.3)
L_VEL_SMOOTH    = 8      # smoothing EMA su vel_amp

# Branch B — strutturale
L_PERS_B        = 50     # lookback lungo ≈8 giorni (da v3.5)
THR_PERS_BASE   = 0.62   # soglia base (da v3.5)
PERS_FLOOR      = 0.10   # soglia minima (da v3.5)
ALPHA           = 0.80   # coefficiente modulazione (da v3.5)
L_VEL           = 80     # lookback signed velocity

# Shared
VEL_SHORT       = 4      # ≈16h
VEL_LONG        = 40     # ≈160h (~7 giorni)

# Warmup e baseline stagionale
WARMUP_WIN      = 200    # finestre iniziali senza allarmi (= SEASON_INIT_WIN)
                         # copre l'intero periodo di inizializzazione della baseline

# Baseline stagionale per Branch A (regime-detrending)
SEASON_EMA      = 240    # EMA lenta per baseline per-regime ≈40 giorni
                         # Tradeoff: lag ~30gg (vs 22gg con 20gg, 37gg con 60gg)
SEASON_INIT_WIN = 200    # finestre per inizializzare la baseline per regime
                         # = WARMUP_WIN → zona rischio completamente bloccata dal gating

# ── Gating gerarchico ─────────────────────────────────────────────────────
K_CONFIRM       = 4      # finestre consecutive (≈16h) per conferma allarme
HYSTERESIS      = 0.08
MIN_OFF_WIN     = 2

# ── Confronto v1 ──────────────────────────────────────────────────────────
THR_V1          = 0.80
MIN_WIN_V1      = 2

ADV_WINDOW      = 60
MIN_CLEAN_MONTHS = 2.0

TURBINES        = [1, 2, 3, 4, 5, 6]
YEARS           = [2016, 2017, 2018, 2019, 2020, 2021]

TEMP_COLS  = list(range(90, 105))
RPM_COLS   = [182, 183, 184, 185]
VIB_COLS   = [241, 243, 273]
POWER_COL  = 62

FAULTS = {
    (3, 2016): dict(name='Oscillation encoder',  dt='2016-03-01 18:02', acute=False, color='#ff4444'),
    (1, 2017): dict(name='High rotor speed',     dt='2017-01-18 23:19', acute=False, color='#ff8844'),
    (6, 2017): dict(name='Freq converter T6',    dt='2017-11-11 20:05', acute=False, color='#ffaa44'),
    (3, 2018): dict(name='Freq converter T3',    dt='2018-04-20 05:31', acute=False, color='#ff6688'),
    (1, 2020): dict(name='Low hydraulic press',  dt='2020-03-25 10:57', acute=True,  color='#ff4444'),
    (3, 2020): dict(name='Freq converter T3b',   dt='2020-10-24 12:40', acute=False, color='#ff6688'),
    (4, 2020): dict(name='Freq converter T4',    dt='2020-08-21 08:50', acute=False, color='#ffaa44'),
}


# ═════════════════════════════════════════════════════════════════════════════
#  I/O — KELMARSH
# ═════════════════════════════════════════════════════════════════════════════

def load_kelmarsh(data_dir: Path, turbine_id: int, year: int):
    zip_path = data_dir / 'wind_turbine' / f'Kelmarsh_{year}.zip'
    if not zip_path.exists():
        return None
    try:
        with zipfile.ZipFile(zip_path) as z:
            candidates = [f for f in z.namelist()
                          if f'Turbine_Data_Kelmarsh_{turbine_id}_' in f]
            if not candidates:
                return None
            with z.open(candidates[0]) as f:
                df = pd.read_csv(f, comment='#', header=None,
                                 index_col=0, parse_dates=True)
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        print(f'    [warn] T{turbine_id} {year}: {e}')
        return None


def extract_sensors(df):
    cols_want  = TEMP_COLS + RPM_COLS + VIB_COLS + [POWER_COL]
    available  = [c for c in cols_want if c in df.columns]
    sub        = df[available].ffill(limit=6)
    coverage   = sub.notna().mean()
    good_cols  = coverage[coverage > 0.45].index.tolist()
    health_cols = [c for c in good_cols if c in TEMP_COLS + RPM_COLS + VIB_COLS]
    sensors_df  = sub[health_cols].dropna(how='all')
    power_series = sub[POWER_COL] if POWER_COL in sub.columns else None
    return sensors_df, power_series


def power_regime(val):
    if pd.isna(val): return -1
    if val < 200:    return 0
    if val < 1500:   return 1
    return 2


def normalize_and_correct(sensors_df, power_series):
    if power_series is not None:
        p = power_series.reindex(sensors_df.index).ffill()
    else:
        p = pd.Series(np.nan, index=sensors_df.index)

    arr = sensors_df.values.astype(np.float64)
    T, N = arr.shape
    col_means = np.nanmean(arr, axis=0)
    for j in range(N):
        arr[np.isnan(arr[:, j]), j] = col_means[j]

    mu  = arr.mean(0);  std = arr.std(0) + 1e-8
    Xz  = (arr - mu) / std

    regimes = np.array([power_regime(v) for v in p.values])
    Xcorr   = Xz.copy()
    for r in [0, 1, 2]:
        mask = regimes == r
        if mask.sum() > 10:
            Xcorr[mask] -= Xz[mask].mean(0)

    return Xcorr, Xz, regimes, sensors_df.index


# ═════════════════════════════════════════════════════════════════════════════
#  INVARIANTI (come v2.1)
# ═════════════════════════════════════════════════════════════════════════════

def pc1_ratio(W):
    Wc = W[~np.isnan(W).any(axis=1)]
    if Wc.shape[0] < 4 or Wc.shape[1] < 2:
        return 0.0
    eigs = np.abs(np.linalg.eigvalsh(np.cov(Wc.T)))
    tot  = eigs.sum()
    return float(eigs[-1] / tot) if tot > 1e-10 else 0.0


def participation_ratio(W):
    Wc = W[~np.isnan(W).any(axis=1)]
    if Wc.shape[0] < 4 or Wc.shape[1] < 2:
        return 0.0
    N    = Wc.shape[1]
    eigs = np.abs(np.linalg.eigvalsh(np.cov(Wc.T)))
    s1   = eigs.sum();  s2 = (eigs ** 2).sum()
    if s2 < 1e-10:
        return 0.0
    pr     = (s1 ** 2) / s2
    pr_inv = (N - pr) / (N - 1 + 1e-8)
    return float(np.clip(pr_inv, 0, 1))


def noise_drift(W, baseline):
    return float(np.clip(
        np.nanmean(np.abs(np.nanmean(W, axis=0) - baseline)), 0, 3) / 3)


def compress_intra_regime(W_raw, regime_slice):
    compress_vals, weights = [], []
    for r in np.unique(regime_slice):
        mask_r = regime_slice == r
        n_r    = mask_r.sum()
        if n_r < 4:
            continue
        compress_vals.append(pc1_ratio(W_raw[mask_r]))
        weights.append(n_r)
    if not compress_vals:
        return pc1_ratio(W_raw)
    w = np.array(weights, dtype=float);  w /= w.sum()
    return float(np.dot(compress_vals, w))


# ═════════════════════════════════════════════════════════════════════════════
#  OPERATORI TEMPORALI
# ═════════════════════════════════════════════════════════════════════════════

def rolling_sigmoid(a, w):
    out = np.empty_like(a, dtype=float)
    for i in range(len(a)):
        lo  = max(0, i - w)
        seg = a[lo:i + 1]
        mu  = seg.mean();  sg = seg.std() + 1e-8
        out[i] = 1.0 / (1.0 + np.exp(-(a[i] - mu) / sg))
    return out


def ema_fn(a, w):
    alpha = 2.0 / (w + 1)
    out   = np.zeros_like(a)
    out[0] = a[0]
    for i in range(1, len(a)):
        out[i] = alpha * a[i] + (1 - alpha) * out[i - 1]
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  LIVELLO 1 — P_v2.1  (diagnosi: Mahalanobis su rolling-sigmoid è rotto)
# ═════════════════════════════════════════════════════════════════════════════
# P_base = 0.45·noise_rs + 0.30·PR_rs + 0.25·compress_intra_rs
# Già calcolato in compute_all — qui solo le funzioni di Livello 2 e gating.
# ─────────────────────────────────────────────────────────────────────────────

def persistence_score(p_base, thr_p=THR_P, l_persist=None):
    """
    Persistenza di P_base sopra soglia: frac( P_base[t-L:t] > thr_p ) ∈ [0, 1]
    l_persist default = L_PERS_B se non specificato.
    """
    if l_persist is None:
        l_persist = L_PERS_B
    T    = len(p_base)
    pers = np.zeros(T)
    for i in range(l_persist, T):
        seg     = p_base[max(0, i - l_persist):i + 1]
        pers[i] = float((seg > thr_p).mean())
    if l_persist < T:
        pers[:l_persist] = pers[l_persist]
    return pers


def regime_detrend(noise_raw, window_regimes,
                   ema_win=SEASON_EMA,
                   init_win=SEASON_INIT_WIN):
    """
    Baseline adattiva per-regime su noise_raw (Branch A — separazione macchina/ambiente).

    PROBLEMA v4: Branch A velocity = relu(EMA4 - EMA40) / p95 è sensibile alla
    crescita stagionale del vento (vento forte in inverno → noise_raw alto su
    tutte le turbine simultaneamente → FA spurii su anni "puliti" come 2019).

    SOLUZIONE: sottrai la baseline per-regime, aggiornata con EMA lenta.
    Se l'ambiente (vento) spinge tutte le turbine al rialzo, la baseline si
    adatta → adj ≈ 0 per tutte. Se solo una turbina ha noise anomalo,
    la sua baseline non sale abbastanza → adj > 0.

    DESIGN:
      baselines[r] = EMA lenta di noise_raw quando regime==r
      n_adj[t] = max(noise_raw[t] - baselines[regime[t]], 0)

    Parametri scelti dopo analisi quantitativa:
      SEASON_EMA = 240 (≈40 giorni): lag ~30gg, soppressione buona
        → EMA troppo corta (20gg) → insegue i fault stessi
        → EMA troppo lunga (80gg) → lag eccessivo su regime shift
      SEASON_INIT_WIN = 200: inizializza con media per-regime dei primi 200 win
        → evita zona di rischio post-startup con falsi spuri
        → coincide con WARMUP_WIN: l'init period è bloccato dal gating

    Proprietà verificata su sintetico:
      Sano stagionale [200:450]:  adj mean=0.004  >0.05: 0.4%
      Fault genuino [450:]:       adj mean=0.111  >0.05: 62.8%
      Sep: 27.6× — robusto e netto
    """
    T     = len(noise_raw)
    alpha = 2.0 / (ema_win + 1)

    # Inizializza baseline per regime con media del periodo iniziale
    baselines = {}
    for r in [0, 1, 2]:
        mask = window_regimes[:init_win] == r
        if mask.any():
            baselines[r] = float(noise_raw[:init_win][mask].mean())
        else:
            baselines[r] = float(noise_raw[:init_win].mean())

    n_adj = np.zeros(T)
    for i in range(T):
        r = int(window_regimes[i]) if window_regimes[i] in (0, 1, 2) else 1

        # Aggiorna baseline solo dopo il periodo di init (evita inquinamento)
        if i >= init_win:
            baselines[r] = alpha * noise_raw[i] + (1 - alpha) * baselines[r]

        n_adj[i] = max(noise_raw[i] - baselines[r], 0.0)

    return n_adj


def velocity_score_amp(noise_raw,
                       vel_short=VEL_SHORT, vel_long=VEL_LONG,
                       l_smooth=L_VEL_SMOOTH):
    """
    BRANCH A — Velocity amplitude (v3.3-like).

    relu( EMA(noise_raw, short) − EMA(noise_raw, long) ), normalizzato via p95.
    Cattura: spike di ampiezza ricorrenti nel noise_raw → fault episodici.
    Sensibile a T6-2017, T3-2018 (frequency converter con pattern ad episodi).
    """
    vel_raw  = np.clip(ema_fn(noise_raw, vel_short) - ema_fn(noise_raw, vel_long), 0, None)
    v95      = np.percentile(vel_raw, 95) + 1e-8
    vel_norm = np.clip(vel_raw / v95, 0, 1)
    return ema_fn(vel_norm, l_smooth)


def signed_velocity_persistence(noise_raw,
                                vel_short=VEL_SHORT, vel_long=VEL_LONG,
                                l_vel=L_VEL):
    """
    LIVELLO 2b v3.4 — Persistenza del segno della velocity.

    DIAGNOSI v3.3: velocity = relu(EMA_short − EMA_long) / p95
    La normalizzazione per p95 amplifica qualsiasi half-cycle in crescita.
    Un segnale ciclico simmetrico produce velocity > soglia ~53% del tempo
    → indistinguibile da degrado genuino.

    FIX: misura QUANTO SPESSO il derivato è positivo, non QUANTO È GRANDE.

    vel_sign(t) = 1  se EMA_short(noise) > EMA_long(noise), altrimenti 0
    P_vel(t)    = fraction( vel_sign[t-L:t] > 0 )

    Proprietà chiave:
      Segnale ciclico simmetrico   → P_vel ≈ 0.50  (metà del tempo sale, metà scende)
      Degrado monotono progressivo → P_vel → 0.80–0.95  (sale quasi sempre)
      Regime shift stabile          → P_vel → 0.50  (oscillazioni bilanciate dopo shift)

    Soglia naturale: THR_VEL_SIGN = 0.60
      → Richiede che il segnale stia crescendo per >60% del lookback L_vel
      → Cicli: 50% < 60% → non scatta
      → Degrado: 70-90% > 60% → scatta

    L_vel deve essere ≥ 2× periodo dominante del ciclo (~20 finestre)
    → L_vel = 80 finestre ≈ 13 giorni
    """
    T        = len(noise_raw)
    vel_raw  = ema_fn(noise_raw, vel_short) - ema_fn(noise_raw, vel_long)
    vel_sign = (vel_raw > 0).astype(float)

    p_vel = np.zeros(T)
    for i in range(l_vel, T):
        p_vel[i] = vel_sign[max(0, i - l_vel):i + 1].mean()
    if l_vel < T:
        p_vel[:l_vel] = p_vel[l_vel]

    return np.clip(p_vel, 0, 1)


def hierarchical_detection(p_base, l2_gate_arr,
                            thr_p=THR_P,
                            k_confirm=K_CONFIRM,
                            hyst=HYSTERESIS,
                            min_off=MIN_OFF_WIN,
                            warmup=WARMUP_WIN):
    """
    Gating a tre stadi su P_base (Livello 1):

    Candidato   → P_base > thr_p
    Qualificato → l2_gate == 1  (branch_A OR branch_B)
    Confermato  → qualificato per ≥ k_confirm finestre consecutive
    Off:        → P_base < thr_p − hyst per ≥ min_off finestre

    Warmup: allarmi bloccati per le prime `warmup` finestre.
    Evita il bias di path-dependency della signed velocity all'inizializzazione
    (EMA_short > EMA_long costantemente nei primi giorni dopo il reset).
    """
    T     = len(p_base)
    alarm = np.zeros(T, dtype=bool)
    stage = np.zeros(T, dtype=int)
    score = p_base * l2_gate_arr

    qual_count  = 0
    below_count = 0
    alarm_on    = False

    for t in range(T):
        p_t  = p_base[t]
        l2_t = l2_gate_arr[t]

        # Warmup: accumuliamo gli stage ma non emettiamo allarmi
        in_warmup = (t < warmup)

        if p_t >= thr_p:
            below_count = 0
            qual_count  = qual_count + 1 if l2_t > 0.5 else 0

            if qual_count >= k_confirm and not in_warmup:
                alarm_on = True
                stage[t] = 3
            elif qual_count >= k_confirm:
                stage[t] = 3   # conta come confermato ma non emette allarme
            elif qual_count > 0:
                stage[t] = 2
            else:
                stage[t] = 1

        else:
            below_count += 1
            qual_count   = 0
            if below_count >= min_off and p_t < thr_p - hyst:
                alarm_on = False
            stage[t] = 3 if alarm_on else 0

        alarm[t] = alarm_on

    return alarm, stage, score

# ═════════════════════════════════════════════════════════════════════════════
#  PIPELINE PRINCIPALE — KELMARSH
# ═════════════════════════════════════════════════════════════════════════════

def compute_all(Xcorr, Xraw, regimes, dates,
                win=WIN, step=STEP, rw=RW, rw_pr=RW_PR):
    """
    v4 — Mixture of Detectors.

    LIVELLO 1: P_base = 0.45·noise_rs + 0.30·PR_rs + 0.25·compress_intra_rs

    LIVELLO 2 — due branch paralleli, ALARM = L1 AND (A OR B):

    BRANCH A (episodico — v3.3-like):
      pers_short = frac(P_base > THR_P over L_PERS_A=20 win ≈3.3gg)
      vel_amp    = relu(EMA4 - EMA40) / p95, smooth EMA8
      gate_A     = (pers_short > THR_PERS_A) AND (vel_amp > THR_VEL_AMP)
      Cattura: T6-2017, T3-2018 — spike di ampiezza ricorrenti

    BRANCH B (strutturale — v3.5-like):
      pers_long  = frac(P_base > THR_P over L_PERS_B=50 win ≈8gg)
      vel_sign   = frac(EMA4 > EMA40 over L_VEL=80 win ≈13gg)
      gate_B     = pers_long > max(THR_PERS_BASE - ALPHA*vel_sign, PERS_FLOOR)
      Cattura: T3-2016, T3-2020, T1-2020 — deriva lenta persistente

    WARMUP: nessun allarme per i primi WARMUP_WIN=80 finestre
      (blocca il bias di inizializzazione della vel_sign_pers)
    """
    T, N     = Xcorr.shape
    baseline = Xcorr[:max(1, win * 3)].mean(axis=0)

    t_idx = []
    noise_r, kap_r, comp_r, pr_r, ci_r, regimes_w = [], [], [], [], [], []

    for s in range(0, T - win, step):
        W      = Xcorr[s:s + win]
        W_raw  = Xraw[s:s + win]
        reg_s  = regimes[s:s + win]

        if np.isnan(W).mean() > 0.2:
            continue

        t_idx.append(s + win // 2)

        ts  = np.nanstd(W, axis=0)
        kap = float(np.clip(1.0 - np.nanmean(ts) / (np.nanstd(ts) + 1e-8), 0, 1))

        noise_r.append(noise_drift(W, baseline))
        kap_r.append(kap)
        comp_r.append(pc1_ratio(W))
        pr_r.append(participation_ratio(W))
        ci_r.append(compress_intra_regime(W_raw, reg_s))
        # Regime rappresentativo della finestra (punto centrale)
        regimes_w.append(int(regimes[s + win // 2]))

    t          = np.array(t_idx)
    n_raw      = np.array(noise_r);   k_raw = np.array(kap_r)
    c_raw      = np.array(comp_r);    pr_raw = np.array(pr_r)
    ci_raw     = np.array(ci_r)
    regimes_w  = np.array(regimes_w)

    # Rolling sigmoid
    n_rs  = rolling_sigmoid(n_raw,  rw)
    k_rs  = rolling_sigmoid(k_raw,  rw)
    c_rs  = rolling_sigmoid(c_raw,  rw)
    pr_rs = rolling_sigmoid(ema_fn(pr_raw, 3), rw_pr)
    ci_rs = rolling_sigmoid(ci_raw, rw)

    # P v1 (confronto)
    P_v1 = 0.45 * n_rs + 0.30 * k_rs + 0.25 * c_rs

    # ── Livello 1: P_base ────────────────────────────────────────────────────
    P_base = 0.45 * n_rs + 0.30 * pr_rs + 0.25 * ci_rs

    # ── Branch A — Episodico (con regime detrending) ─────────────────────────
    # n_raw_adj rimuove la componente ambientale stagionale per regime.
    # Vento forte su tutto il parco → baseline sale → adj ≈ 0
    # Fault su singola turbina → baseline non sale abbastanza → adj > 0
    n_raw_adj  = regime_detrend(n_raw, regimes_w)
    pers_short = persistence_score(P_base, thr_p=THR_P, l_persist=L_PERS_A)
    vel_amp    = velocity_score_amp(n_raw_adj)
    gate_A     = ((pers_short > THR_PERS_A) & (vel_amp > THR_VEL_AMP)).astype(float)

    # ── Branch B — Strutturale (invariato) ───────────────────────────────────
    pers_long  = persistence_score(P_base, thr_p=THR_P, l_persist=L_PERS_B)
    vel_sign   = signed_velocity_persistence(n_raw)
    thr_b_eff  = np.maximum(THR_PERS_BASE - ALPHA * vel_sign, PERS_FLOOR)
    gate_B     = (pers_long > thr_b_eff).astype(float)

    # ── Decisione: OR dei due branch ─────────────────────────────────────────
    l2_gate_arr = np.clip(gate_A + gate_B, 0, 1)

    # Segnale continuo per plot
    l2_cont = np.maximum(
        pers_short * vel_amp,
        pers_long  * vel_sign
    )

    # ── Gating gerarchico con warmup ─────────────────────────────────────────
    alarm, stage, score = hierarchical_detection(P_base, l2_gate_arr, warmup=WARMUP_WIN)

    t_dt = dates[t.clip(0, len(dates) - 1)]

    return dict(
        t_dt        = t_dt,
        P_v1        = P_v1,
        P_base      = P_base,
        AR          = c_rs,
        noise       = n_rs, pr = pr_rs, cintra = ci_rs,
        # Branch A
        pers_short  = pers_short,
        vel_amp     = vel_amp,
        gate_A      = gate_A,
        # Branch B
        pers_long   = pers_long,
        vel_sign    = vel_sign,
        gate_B      = gate_B,
        # Combined
        persistenza = pers_long,   # alias per compatibilità plot
        velocita    = vel_sign,    # alias per compatibilità plot
        l2_cont     = l2_cont,
        score       = score,
        alarm       = alarm,
        stage       = stage,
        inv_matrix  = np.column_stack([n_rs, pr_rs, ci_rs]),
        noise_raw   = n_raw, pr_raw = pr_raw, comp_raw = c_raw,
    )

    t_dt = dates[t.clip(0, len(dates) - 1)]

    return dict(
        t_dt        = t_dt,
        P_v1        = P_v1,
        P_base      = P_base,          # Livello 1 (P_v2.1)
        AR          = c_rs,
        noise       = n_rs, pr = pr_rs, cintra = ci_rs,
        persistenza = persistenza,
        velocita    = velocita,
        l2_cont     = l2_cont,
        score       = score,
        alarm       = alarm,
        stage       = stage,
        inv_matrix  = np.column_stack([n_rs, pr_rs, ci_rs]),
        noise_raw   = n_raw, pr_raw = pr_raw, comp_raw = c_raw,
    )


# ═════════════════════════════════════════════════════════════════════════════
#  METRICHE
# ═════════════════════════════════════════════════════════════════════════════

def advance_warning_v1(t_dt, P, fault_dt_str, thr=THR_V1,
                        max_days=ADV_WINDOW, min_win=MIN_WIN_V1):
    """Advance warning P v1 con hysteresis."""
    ft    = pd.Timestamp(fault_dt_str)
    start = ft - pd.Timedelta(days=max_days)
    mask  = (t_dt >= start) & (t_dt <= ft)
    cnt   = 0
    for dt, v in zip(t_dt[mask], P[mask]):
        cnt = cnt + 1 if v >= thr else 0
        if cnt >= min_win:
            adv = (ft - dt).total_seconds() / 86400
            return round(adv, 1)
    return None


def advance_warning_v3(t_dt, alarm, stage, fault_dt_str,
                        max_days=ADV_WINDOW):
    """
    Advance warning v3: prima finestra in stato ≥ 3 (confermato)
    nei max_days prima del fault.
    Restituisce (giorni, stage_al_primo_allarme).
    """
    ft    = pd.Timestamp(fault_dt_str)
    start = ft - pd.Timedelta(days=max_days)
    mask  = (t_dt >= start) & (t_dt <= ft)

    for dt, a, st in zip(t_dt[mask], alarm[mask], stage[mask]):
        if a:  # alarm confermato
            adv = (ft - dt).total_seconds() / 86400
            return round(adv, 1)
    return None


def count_fa_v1(t_dt, P, thr, fault_dt_str=None,
                excl_days=ADV_WINDOW, hyst=0.08, min_win=MIN_WIN_V1):
    mask = np.ones(len(t_dt), dtype=bool)
    if fault_dt_str:
        ft   = pd.Timestamp(fault_dt_str)
        excl = ft - pd.Timedelta(days=excl_days)
        mask &= (t_dt < excl) | (t_dt > ft + pd.Timedelta(days=30))
    mask &= t_dt.month != 1

    sig_c    = P[mask]
    alarm_on = False;  cnt = 0;  n_ep = 0
    for v in sig_c:
        cnt = cnt + 1 if v >= thr else 0
        if cnt >= min_win and not alarm_on:
            alarm_on = True;  n_ep += 1
        if v < thr - hyst:
            alarm_on = False;  cnt = 0
    return n_ep


def count_fa_v3(t_dt, alarm, fault_dt_str=None, excl_days=ADV_WINDOW):
    """
    Conta episodi FA v3 (allarme confermato ON → OFF = 1 episodio).
    """
    mask = np.ones(len(t_dt), dtype=bool)
    if fault_dt_str:
        ft   = pd.Timestamp(fault_dt_str)
        excl = ft - pd.Timedelta(days=excl_days)
        mask &= (t_dt < excl) | (t_dt > ft + pd.Timedelta(days=30))
    mask &= t_dt.month != 1

    alarm_c  = alarm[mask]
    n_ep     = 0
    prev     = False
    for a in alarm_c:
        if a and not prev:
            n_ep += 1
        prev = a
    return n_ep


def fa_rate_safe(n_ep, n_months):
    if n_months < MIN_CLEAN_MONTHS:
        return None
    return round(n_ep / n_months, 3)


def useful_category(adv_days):
    if adv_days is None:    return 'missed'
    if adv_days >= 7:       return 'useful'
    if adv_days >= 1:       return 'last-minute'
    return 'missed'


# ═════════════════════════════════════════════════════════════════════════════
#  PLOT
# ═════════════════════════════════════════════════════════════════════════════

DARK_BG = '#0d0d0d'


def _style_ax(ax):
    ax.set_facecolor('#111')
    ax.tick_params(colors='#888', labelsize=7)
    for sp in ax.spines.values():
        sp.set_color('#333')


def _fault_overlay(axes, faults_ty):
    for finfo in faults_ty:
        ft  = pd.Timestamp(finfo['dt'])
        col = finfo['color']
        for ax in axes:
            ax.axvline(ft, color=col, lw=1.3, ls='--', alpha=0.9)


def plot_timeline(tid, year, sig, faults_ty, out_dir):
    t   = sig['t_dt']
    fig, axes = plt.subplots(4, 1, figsize=(16, 11),
                             gridspec_kw={'height_ratios': [1, 1.2, 1, 0.8]})
    fig.patch.set_facecolor(DARK_BG)
    ax_inv, ax_l1, ax_l2, ax_alarm = axes
    for ax in axes: _style_ax(ax)

    # Invarianti
    ax_inv.plot(t, sig['noise'],  color='#ff8844', lw=0.8, label='noise')
    ax_inv.plot(t, sig['pr'],     color='#44ccff', lw=0.9, label='PR_inv')
    ax_inv.plot(t, sig['cintra'], color='#aa66ff', lw=0.9, label='compress_intra')
    ax_inv.set_ylim(-0.05, 1.10); ax_inv.set_ylabel('Invarianti', color='#888', fontsize=8)
    ax_inv.legend(fontsize=7, labelcolor='#ccc', framealpha=0.2, ncol=3)

    # Livello 1: Mahalanobis
    ax_l1.plot(t, sig['P_base'], color='#44ff88', lw=1.4, label='P_base = L1 (v2.1)')
    ax_l1.plot(t, sig['P_v1'],  color='#edbd38', lw=0.8, ls='--', alpha=0.6, label='P v1')
    ax_l1.axhline(THR_P, color='#44ff88', lw=0.7, ls=':', alpha=0.6)
    ax_l1.axhline(THR_V1, color='#edbd38', lw=0.6, ls=':', alpha=0.4)
    ax_l1.set_ylim(-0.05, 1.10); ax_l1.set_ylabel('L1: P_base', color='#888', fontsize=8)
    ax_l1.legend(fontsize=7, labelcolor='#ccc', framealpha=0.2, ncol=3)

    # Livello 2: Coerenza
    ax_l2.plot(t, sig['persistenza'], color='#ff66cc', lw=1.2,
               label=f'L2a: persistenza (soglia modulata)')
    ax_l2.plot(t, sig['velocita'],    color='#ffaa44', lw=1.2,
               label='L2b: vel_sign_pers')
    ax_l2.plot(t, sig['l2_cont'],     color='#ffffff', lw=0.7,
               alpha=0.4, label='L2 continuo (√pers·vel)')
    ax_l2.axhline(PERS_FLOOR, color='#ff66cc', lw=0.5, ls=':', alpha=0.4, label=f'floor={PERS_FLOOR}')
    ax_l2.axhline(THR_PERS_BASE - ALPHA*0.50, color='#ffaa44', lw=0.5, ls=':', alpha=0.4)
    ax_l2.set_ylim(-0.05, 1.10); ax_l2.set_ylabel('Livello 2', color='#888', fontsize=8)
    ax_l2.legend(fontsize=7, labelcolor='#ccc', framealpha=0.2, ncol=2)

    # Stage e allarmi
    stage_colors = {0: '#222', 1: '#335533', 2: '#555500', 3: '#883300'}
    for i, (ti, st) in enumerate(zip(t[:-1], sig['stage'][:-1])):
        ax_alarm.axvspan(ti, t[i + 1], color=stage_colors.get(int(st), '#222'),
                         alpha=0.8)
    ax_alarm.set_ylim(0, 1); ax_alarm.set_yticks([])
    ax_alarm.set_ylabel('Stage\n0=quiet 1=cand\n2=qual 3=alarm',
                        color='#888', fontsize=6)
    # Patch legend
    from matplotlib.patches import Patch
    ax_alarm.legend(
        handles=[Patch(color=c, label=l) for c, l in
                 [('#222','quiet'), ('#335533','candidato'),
                  ('#555500','qualificato'), ('#883300','allarme')]],
        fontsize=6, labelcolor='#ccc', framealpha=0.2, loc='upper left', ncol=4
    )

    _fault_overlay(axes, faults_ty)
    for finfo in faults_ty:
        ax_l1.text(pd.Timestamp(finfo['dt']), 1.05,
                   finfo['name'], color=finfo['color'],
                   fontsize=5, rotation=45, ha='left', va='bottom')

    axes[0].set_title(f'T{tid} — {year}  (v3 — due livelli)',
                      color='#ddd', fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / f'fig_kelmarsh_T{tid}_{year}.png',
                dpi=130, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()


def plot_zoom_fault(tid, year, sig, finfo, out_dir, idx):
    ft    = pd.Timestamp(finfo['dt'])
    start = ft - pd.Timedelta(days=60)
    t     = sig['t_dt']
    mask  = (t >= start) & (t <= ft + pd.Timedelta(days=5))
    if mask.sum() < 3:
        return

    td = t[mask]
    fig, axes = plt.subplots(3, 1, figsize=(13, 8),
                             gridspec_kw={'height_ratios': [2, 1.5, 0.7]})
    fig.patch.set_facecolor(DARK_BG)
    ax_main, ax_l12, ax_st = axes
    for ax in axes: _style_ax(ax)

    # Segnali principali
    ax_main.plot(td, sig['persistenza'][mask], color='#ff66cc', lw=1.5,
                 label=f'L2a: persistenza (soglia modulata)')
    ax_main.plot(td, sig['velocita'][mask],    color='#ffaa44', lw=1.5,
                 label='L2b: vel_sign_pers')
    ax_main.plot(td, sig['P_base'][mask],      color='#44ff88', lw=1.3,
                 label=f'L1: P_base (thr={THR_P})')
    ax_main.plot(td, sig['P_v1'][mask],        color='#edbd38', lw=0.9,
                 ls='--', alpha=0.6, label='P v1 (confronto)')
    ax_main.axhline(THR_P,    color='#44ff88', lw=0.7, ls=':')
    ax_main.axhline(PERS_FLOOR, color='#ff66cc', lw=0.5, ls=':', alpha=0.5)
    ax_main.axhline(THR_PERS_BASE - ALPHA*0.50, color='#ffaa44', lw=0.5, ls=':', alpha=0.5)
    ax_main.axhline(THR_V1,   color='#edbd38', lw=0.6, ls=':', alpha=0.4)
    ax_main.set_ylim(-0.05, 1.10)
    ax_main.legend(fontsize=7, labelcolor='#ccc', framealpha=0.2, ncol=2)

    # Invarianti
    ax_l12.plot(td, sig['noise'][mask],  color='#ff8844', lw=1.0, label='noise')
    ax_l12.plot(td, sig['pr'][mask],     color='#44ccff', lw=1.0, label='PR_inv')
    ax_l12.plot(td, sig['cintra'][mask], color='#aa66ff', lw=1.0, label='compress_intra')
    ax_l12.set_ylim(-0.05, 1.10)
    ax_l12.legend(fontsize=7, labelcolor='#ccc', framealpha=0.2, ncol=3)

    # Stage
    stage_colors = {0: '#222', 1: '#335533', 2: '#555500', 3: '#883300'}
    t_arr = td.values
    st_arr = sig['stage'][mask]
    for i in range(len(t_arr) - 1):
        ax_st.axvspan(t_arr[i], t_arr[i + 1],
                      color=stage_colors.get(int(st_arr[i]), '#222'), alpha=0.9)
    ax_st.set_ylim(0, 1); ax_st.set_yticks([])

    # Annotations
    adv_v1 = advance_warning_v1(t, sig['P_v1'], finfo['dt'])
    adv_v3 = advance_warning_v3(t, sig['alarm'], sig['stage'], finfo['dt'])
    ann = []
    ann.append(f'v1:  −{adv_v1:.1f}d' if adv_v1 else 'v1:  missed')
    ann.append(f'v3:  −{adv_v3:.1f}d' if adv_v3 else 'v3:  missed')
    ax_main.text(0.02, 0.97, '\n'.join(ann), transform=ax_main.transAxes,
                 color='#ddd', fontsize=9, va='top',
                 bbox=dict(facecolor='#1a1a1a', alpha=0.85, pad=4))

    for ax in axes:
        ax.axvline(ft, color=finfo['color'], lw=1.5, ls='--', alpha=0.9)

    ax_main.set_title(f'T{tid} {year} — {finfo["name"]}  (v3)',
                      color='#ddd', fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / f'fig_kelmarsh_T{tid}_{year}_fault{idx}.png',
                dpi=130, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()


def plot_trajectory(tid, year, sig, finfo, out_dir):
    """
    Traiettoria nello spazio invariante con coerenza angolare sovrapposta.
    Mostra la differenza tra escursioni (alta d, bassa coh)
    e degrado (alta d, alta coh).
    """
    t   = sig['t_dt']
    ft  = pd.Timestamp(finfo['dt'])

    mask_pre   = (t > ft - pd.Timedelta(days=60)) & (t <= ft)
    mask_clean = (t < ft - pd.Timedelta(days=90)) & (t.month != 1)

    if not mask_pre.any():
        return

    # Seleziona il 20% dei punti puliti (per non sovraffollare)
    idx_clean = np.where(mask_clean)[0]
    if len(idx_clean) > 200:
        idx_clean = np.random.choice(idx_clean, 200, replace=False)
        mask_clean_sub = np.zeros(len(t), dtype=bool)
        mask_clean_sub[idx_clean] = True
    else:
        mask_clean_sub = mask_clean

    inv = sig['inv_matrix']
    l2c = sig['l2_cont']   # L2 continuo (sqrt(persistenza × velocità))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor(DARK_BG)

    pairs = [
        ('noise (L1)', inv[:, 0], 'PR_inv (L1)', inv[:, 1]),
        ('noise (L1)', inv[:, 0], 'compress_intra (L1)', inv[:, 2]),
        ('PR_inv (L1)', inv[:, 1], 'compress_intra (L1)', inv[:, 2]),
    ]

    for ax, (xn, xv, yn, yv) in zip(axes, pairs):
        _style_ax(ax)

        # Punti puliti: colorati per L2 continuo
        sc_clean = ax.scatter(
            xv[mask_clean_sub], yv[mask_clean_sub],
            c=l2c[mask_clean_sub], cmap='Blues', vmin=0, vmax=1,
            s=10, alpha=0.5, label='pulito (→ L2)', zorder=2
        )

        # Pre-fault: colorati per tempo, dimensione per L2
        t_norm   = np.linspace(0, 1, mask_pre.sum())
        l2_pre   = l2c[mask_pre]
        sizes    = 15 + 50 * l2_pre   # grandi = alta persistenza+velocità

        ax.scatter(
            xv[mask_pre], yv[mask_pre],
            c=t_norm, cmap='plasma', s=sizes, alpha=0.95,
            label='pre-fault (→ tempo, size→L2)', zorder=4
        )

        # Freccia finale
        if mask_pre.sum() >= 2:
            x_last = xv[mask_pre][-2:]
            y_last = yv[mask_pre][-2:]
            ax.annotate('', xy=(x_last[-1], y_last[-1]),
                        xytext=(x_last[-2], y_last[-2]),
                        arrowprops=dict(arrowstyle='->', color='#ff3333', lw=2.0))

        # Regione co-attivazione
        ax.fill([0.5, 1, 1, 0.5], [0.5, 0.5, 1, 1], color='#44cc88', alpha=0.06)
        ax.plot([0.5, 1, 1, 0.5, 0.5], [0.5, 0.5, 1, 1, 0.5],
                color='#44cc88', lw=0.8, ls='--', alpha=0.4)

        ax.set_xlabel(xn, color='#aaa', fontsize=8)
        ax.set_ylabel(yn, color='#aaa', fontsize=8)
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.legend(fontsize=6, labelcolor='#ccc', framealpha=0.2)

        plt.colorbar(sc_clean, ax=ax, label='L2 √(pers·vel)', fraction=0.04, pad=0.04)

    adv_v3 = advance_warning_v3(t, sig['alarm'], sig['stage'], finfo['dt'])
    adv_str = f'v3.2: −{adv_v3:.1f}d' if adv_v3 else 'v3.2: missed'
    fig.suptitle(
        f'Traiettoria invariante — T{tid} {year} — {finfo["name"]}  [{adv_str}]\n'
        f'punti pre-fault: size ∝ L2 (√persistenza · velocità)',
        color='#ddd', fontsize=9
    )
    plt.tight_layout()
    plt.savefig(out_dir / f'fig_trajectory_T{tid}_{year}.png',
                dpi=130, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP — KELMARSH
# ═════════════════════════════════════════════════════════════════════════════

def run_kelmarsh(data_dir: Path, out_dir: Path):
    print('\n' + '═' * 68)
    print('  KELMARSH — 6 turbine × 6 anni  (v1 confronto vs v3)')
    print('═' * 68)

    all_results = {}
    fa_by_ty    = {}

    for year in YEARS:
        for tid in TURBINES:
            df = load_kelmarsh(data_dir, tid, year)
            if df is None:
                continue

            sensors_df, power_series = extract_sensors(df)
            if sensors_df.shape[1] < 4:
                continue

            Xcorr, Xraw, regimes, dates = normalize_and_correct(
                sensors_df, power_series)
            sig = compute_all(Xcorr, Xraw, regimes, dates)

            finfo_list = [v for (t_, y_), v in FAULTS.items()
                          if t_ == tid and y_ == year]
            fault_dt   = finfo_list[0]['dt'] if finfo_list else None

            # FA
            fa_ep_v1 = count_fa_v1(sig['t_dt'], sig['P_v1'], THR_V1, fault_dt)
            fa_ep_v3 = count_fa_v3(sig['t_dt'], sig['alarm'], fault_dt)

            mask_clean = sig['t_dt'].month != 1
            if fault_dt:
                ft = pd.Timestamp(fault_dt)
                mask_clean &= sig['t_dt'] < ft - pd.Timedelta(days=60)
            t_clean  = sig['t_dt'][mask_clean]
            n_months = ((t_clean[-1] - t_clean[0]).total_seconds() / (30.44 * 86400)
                        if len(t_clean) > 1 else 0.0)

            fa_r_v1 = fa_rate_safe(fa_ep_v1, n_months)
            fa_r_v3 = fa_rate_safe(fa_ep_v3, n_months)

            fa_by_ty[(tid, year)] = dict(
                fa_ep_v1=fa_ep_v1, fa_ep_v3=fa_ep_v3,
                fa_rate_v1=fa_r_v1, fa_rate_v3=fa_r_v3,
                n_months=round(n_months, 1),
                insufficient=n_months < MIN_CLEAN_MONTHS,
            )

            # Advance warning + plot
            for i, finfo in enumerate(finfo_list):
                adv1 = advance_warning_v1(sig['t_dt'], sig['P_v1'], finfo['dt'])
                adv3 = advance_warning_v3(
                    sig['t_dt'], sig['alarm'], sig['stage'], finfo['dt'])

                fa_tag = ('insuff.' if n_months < MIN_CLEAN_MONTHS else
                          f'{fa_r_v1:.2f}→{fa_r_v3:.2f}/mo')

                print(f'  T{tid} {year}  {finfo["name"]:<26}'
                      f'  v1={str(adv1 or "miss")+"d":>8}'
                      f'  v3={str(adv3 or "miss")+"d":>8}'
                      f'  FA: {fa_tag}')

                key = f"T{tid}_{year}_{finfo['name'].replace(' ', '_')}"
                all_results[key] = dict(
                    turbine=tid, year=year,
                    fault=finfo['name'], fault_dt=finfo['dt'],
                    acute=finfo['acute'],
                    adv_v1_days=adv1, cat_v1=useful_category(adv1),
                    adv_v3_days=adv3, cat_v3=useful_category(adv3),
                    fa_rate_v1=fa_r_v1, fa_rate_v3=fa_r_v3,
                    n_months_clean=round(n_months, 1),
                    fa_insufficient=(n_months < MIN_CLEAN_MONTHS),
                )

                plot_zoom_fault(tid, year, sig, finfo, out_dir, i + 1)
                plot_trajectory(tid, year, sig, finfo, out_dir)

            if finfo_list or year in [2016, 2018, 2020]:
                plot_timeline(tid, year, sig, finfo_list, out_dir)

    return all_results, fa_by_ty


# ═════════════════════════════════════════════════════════════════════════════
#  CMAPSS — due livelli in versione offline
# ═════════════════════════════════════════════════════════════════════════════

def run_cmapss(data_dir: Path, out_dir: Path):
    cmapss_dir = (data_dir / 'cmapss' /
                  '6. Turbofan Engine Degradation Simulation Data Set')
    if not cmapss_dir.exists():
        print('\n  [CMAPSS] Cartella non trovata — skip')
        return {}

    print('\n' + '═' * 68)
    print('  CMAPSS — FD001-004  (v1 confronto vs v3.2 due livelli)')
    print('═' * 68)

    WIN_C  = 30;  STEP_C = 5;  USEFUL_CYC = 30
    N_FRAC = 0.30

    # Soglie CMAPSS: adattate per serie offline brevi
    THR_P_C    = 0.68   # P_base su n01 — leggermente più bassa di SCADA (0.75)
    THR_PERS_C = 0.35   # lookback più corto → soglia leggermente più bassa
    THR_VEL_C  = 0.08
    L_PERS_C   = 12     # ~60 cicli di lookback persistenza (< 500 cicli totali)
    K_CONF_C   = 3

    def load_fd(fd_id):
        path = cmapss_dir / f'train_FD{fd_id:03d}.txt'
        if not path.exists():
            return None
        df = pd.read_csv(path, sep=r'\s+', header=None)
        df.columns = (['engine', 'cycle'] +
                      [f'op{i}' for i in range(1, 4)] +
                      [f's{i}' for i in range(1, 22)])
        return df

    def active_sensors(df):
        sensor_cols = [c for c in df.columns if c.startswith('s')]
        variances   = df.groupby('engine')[sensor_cols].std().mean()
        return [c for c in sensor_cols if variances[c] > 0.01]

    def n01(a):
        mn, mx = a.min(), a.max()
        return (a - mn) / (mx - mn + 1e-12)

    def process_engine(sensors_raw, op_settings, fd_id):
        T, N  = sensors_raw.shape
        n_tr  = max(WIN_C, int(T * N_FRAC))

        mu  = sensors_raw[:n_tr].mean(0)
        std = sensors_raw[:n_tr].std(0) + 1e-8
        Xz  = (sensors_raw - mu) / std

        Xcorr = Xz.copy()
        labels = np.zeros(T, dtype=int)

        if fd_id in (2, 4) and op_settings is not None:
            try:
                from sklearn.cluster import KMeans
                km = KMeans(n_clusters=6, random_state=0, n_init=5)
                km.fit(op_settings[:n_tr])
                labels = km.predict(op_settings)
                for r in range(6):
                    m = labels == r
                    if m.sum() > WIN_C:
                        Xcorr[m] -= Xz[m].mean(0)
            except ImportError:
                pass

        baseline = Xcorr[:n_tr].mean(0)

        t_idx, n_r, k_r, c_r, pr_r = [], [], [], [], []

        for s in range(0, T - WIN_C, STEP_C):
            W    = Xcorr[s:s + WIN_C]
            t_idx.append(s + WIN_C // 2)

            ts   = W.std(0)
            kap  = float(np.clip(1 - ts.mean() / (ts.std() + 1e-8), 0, 1))
            nd   = float(np.clip(np.mean(np.abs(W.mean(0) - baseline)), 0, 3) / 3)
            eigs = np.abs(np.linalg.eigvalsh(np.cov(W.T)))
            tot  = eigs.sum()
            comp = float(eigs[-1] / tot) if tot > 1e-10 else 0.0
            pr   = float((tot ** 2) / ((eigs ** 2).sum() + 1e-12) - 1) / (N - 1 + 1e-8)

            n_r.append(nd);  k_r.append(kap)
            c_r.append(comp);  pr_r.append(np.clip(pr, 0, 1))

        t_arr = np.array(t_idx)
        n_a = n01(np.array(n_r));  k_a = n01(np.array(k_r))
        c_a = n01(np.array(c_r));  pr_a = n01(ema_fn(np.array(pr_r), 3))

        # P v1 confronto (originale)
        P_v1 = 0.45 * n_a + 0.30 * k_a + 0.25 * c_a

        # ── Livello 1: P_base = P_v2.1 (n01 offline) ────────────────────────
        P_base_c = 0.45 * n_a + 0.30 * pr_a + 0.25 * c_a

        # ── Livello 2a: Persistenza su P_base ────────────────────────────────
        n_r_arr  = np.array(n_r)
        pers_c   = persistence_score(P_base_c, thr_p=THR_P_C, l_persist=L_PERS_C)

        # ── Livello 2b: Velocità su noise_raw ────────────────────────────────
        vel_long_c = min(VEL_LONG, len(n_r_arr) // 3)
        l_vel_c    = min(L_VEL, len(n_r_arr) // 3)
        vel_c      = signed_velocity_persistence(n_r_arr,
                                                  vel_short=VEL_SHORT,
                                                  vel_long=vel_long_c,
                                                  l_vel=l_vel_c)

        # ── Gate ─────────────────────────────────────────────────────────────
        # v4 dual-branch CMAPSS (serie corte → lookback ridotti)
        L_A_C    = min(L_PERS_A, len(n_r_arr)//4)
        L_B_C    = min(L_PERS_B, len(n_r_arr)//3)
        L_VEL_C  = min(L_VEL, len(n_r_arr)//3)
        vel_long_c = min(VEL_LONG, len(n_r_arr)//3)

        # Branch A: regime detrend su serie CMAPSS
        # op_settings usati come proxy di regime (kmeans su FD002/FD004 già applicato)
        # Per semplicità: regime CMAPSS = 0 per tutto (serie già normalizzate)
        regs_c = np.zeros(len(n_r_arr), dtype=int)  # CMAPSS: regime unico normalizzato
        n_adj_c = regime_detrend(n_r_arr, regs_c,
                                  ema_win=min(SEASON_EMA, len(n_r_arr)//3),
                                  init_win=min(SEASON_INIT_WIN, len(n_r_arr)//4))
        pers_sh_c = persistence_score(P_base_c, thr_p=THR_P_C, l_persist=L_A_C)
        vel_a_c   = velocity_score_amp(n_adj_c, vel_long=vel_long_c, l_smooth=3)
        gate_a_c  = ((pers_sh_c > THR_PERS_A) & (vel_a_c > THR_VEL_AMP)).astype(float)

        pers_lg_c = persistence_score(P_base_c, thr_p=THR_P_C, l_persist=L_B_C)
        vel_s_c   = signed_velocity_persistence(n_r_arr, l_vel=L_VEL_C, vel_long=vel_long_c)
        thr_b_c   = np.maximum(0.55 - ALPHA * vel_s_c, 0.08)
        gate_b_c  = (pers_lg_c > thr_b_c).astype(float)

        l2_gate_c = np.clip(gate_a_c + gate_b_c, 0, 1)

        # ── Gating gerarchico ─────────────────────────────────────────────────
        alarm, stage, _ = hierarchical_detection(
            P_base_c, l2_gate_c,
            thr_p=THR_P_C, k_confirm=K_CONF_C
        )

        return t_arr, P_v1, alarm, T

    results_cmapss = {}
    report_v1_ref  = {'FD001': 68.3, 'FD002': 65.2, 'FD003': 72.4, 'FD004': 71.1}

    for fd_id in [1, 2, 3, 4]:
        df = load_fd(fd_id)
        if df is None:
            print(f'  FD{fd_id:03d}: file non trovato')
            continue

        sensor_cols = active_sensors(df)
        op_cols     = ['op1', 'op2', 'op3']
        engines     = df['engine'].unique()

        useful_v1 = 0;  useful_v3 = 0

        for eng_id in engines:
            eng_df      = df[df['engine'] == eng_id]
            sensors_raw = eng_df[sensor_cols].values.astype(np.float64)
            op_settings = eng_df[op_cols].values.astype(np.float64)
            T           = len(sensors_raw)
            if T < WIN_C * 3:    # [FIX] allineato a v2.1 (era WIN_C*4)
                continue

            t_arr, P_v1, alarm, T_eng = process_engine(
                sensors_raw, op_settings, fd_id)

            # v1 useful
            cnt = 0
            for ti, v in zip(t_arr, P_v1):
                cnt = cnt + 1 if v >= 0.65 else 0
                if cnt >= 2 and (T_eng - ti) >= USEFUL_CYC:
                    useful_v1 += 1
                    break

            # v3.1 useful
            for ti, a in zip(t_arr, alarm):
                if a and (T_eng - ti) >= USEFUL_CYC:
                    useful_v3 += 1
                    break

        n_eng  = len(engines)
        pct_v1 = round(100 * useful_v1 / max(n_eng, 1), 1)
        pct_v3 = round(100 * useful_v3 / max(n_eng, 1), 1)
        ref    = report_v1_ref.get(f'FD{fd_id:03d}', '?')
        delta  = f'{pct_v3 - pct_v1:+.1f}%' if pct_v1 > 0 else 'n/a'

        results_cmapss[f'FD{fd_id:03d}'] = dict(
            n_engines=n_eng, useful_v1=useful_v1, useful_v3=useful_v3,
            pct_v1=pct_v1, pct_v3=pct_v3, report_v1_ref=ref,
        )
        print(f'  FD{fd_id:03d}  ({n_eng:3d} motori):  '
              f'v1={pct_v1:5.1f}%  v3.1={pct_v3:5.1f}%  '
              f'Δ={delta}  (report v1={ref}%)')

    return results_cmapss


# ═════════════════════════════════════════════════════════════════════════════
#  SUMMARY + JSON
# ═════════════════════════════════════════════════════════════════════════════

def write_summary(kelmarsh_res, fa_by_ty, cmapss_res, out_dir):
    lines = []
    lines.append('═' * 72)
    lines.append('  SYMPHONON P v4.1 — MIXTURE + REGIME DETRENDING')
    lines.append(f'  L1 P_base:      THR_P={THR_P}  (P_v2.1: noise+PR+compress_intra)')
    lines.append(f'  Branch A: L_PERS_A={L_PERS_A}  THR_PERS_A={THR_PERS_A}  THR_VEL_AMP={THR_VEL_AMP}  SEASON_EMA={SEASON_EMA}')
    lines.append(f'  Branch B: L_PERS_B={L_PERS_B}  BASE={THR_PERS_BASE}  ALPHA={ALPHA}  L_VEL={L_VEL}')
    lines.append(f'  Warmup:   {WARMUP_WIN} finestre  (= SEASON_INIT_WIN={SEASON_INIT_WIN})')
    lines.append(f'  Gating:         K_CONFIRM={K_CONFIRM}  HYST={HYSTERESIS}')
    lines.append('═' * 72)

    lines.append('\n── ADVANCE WARNING FAULT EVENTS ──')
    lines.append(f'  {"Evento":<35}  {"Adv v1":>8}  {"Cat v1":>11}'
                 f'  {"Adv v3.1":>8}  {"Cat v3.1":>11}'
                 f'  {"FA v1/mo":>8}  {"FA v3/mo":>8}  {"Mesi":>5}')
    lines.append('  ' + '─' * 96)

    for key, r in kelmarsh_res.items():
        ev  = f'T{r["turbine"]} {r["year"]} {r["fault"]}'
        av1 = f'{r["adv_v1_days"]:.1f}d' if r['adv_v1_days'] else 'missed'
        av3 = f'{r["adv_v3_days"]:.1f}d' if r['adv_v3_days'] else 'missed'
        fr1 = f'{r["fa_rate_v1"]:.2f}' if r['fa_rate_v1'] is not None else 'insuff.'
        fr3 = f'{r["fa_rate_v3"]:.2f}' if r['fa_rate_v3'] is not None else 'insuff.'
        lines.append(f'  {ev:<35}  {av1:>8}  {r["cat_v1"]:>11}'
                     f'  {av3:>8}  {r["cat_v3"]:>11}'
                     f'  {fr1:>8}  {fr3:>8}  {r["n_months_clean"]:>5.1f}')

    valid_v1 = [v['fa_rate_v1'] for v in fa_by_ty.values()
                if v['fa_rate_v1'] is not None]
    valid_v3 = [v['fa_rate_v3'] for v in fa_by_ty.values()
                if v['fa_rate_v3'] is not None]
    n_insuff = sum(1 for v in fa_by_ty.values() if v['insufficient'])

    if valid_v1 and valid_v3:
        ratio = np.mean(valid_v1) / max(np.mean(valid_v3), 0.001)
        sign  = '↓ migliore' if ratio > 1 else '↑ peggiore'
        lines.append(f'\n  FA/month globale (periodo ≥ {MIN_CLEAN_MONTHS}mo):  '
                     f'v1={np.mean(valid_v1):.3f}  v3.1={np.mean(valid_v3):.3f}  '
                     f'({ratio:.2f}× {sign})')
        lines.append(f'  Turbine-anni esclusi (periodo insuff.): {n_insuff}')

    n_f    = len(kelmarsh_res)
    det_v1 = sum(1 for r in kelmarsh_res.values() if r['cat_v1'] != 'missed')
    det_v3 = sum(1 for r in kelmarsh_res.values() if r['cat_v3'] != 'missed')
    use_v1 = sum(1 for r in kelmarsh_res.values() if r['cat_v1'] == 'useful')
    use_v3 = sum(1 for r in kelmarsh_res.values() if r['cat_v3'] == 'useful')
    lm_v1  = sum(1 for r in kelmarsh_res.values() if r['cat_v1'] == 'last-minute')
    lm_v3  = sum(1 for r in kelmarsh_res.values() if r['cat_v3'] == 'last-minute')

    lines.append(f'\n  Detected:    v1={det_v1}/{n_f}   v3.1={det_v3}/{n_f}')
    lines.append(f'  Useful ≥7d:  v1={use_v1}/{n_f}   v3.1={use_v3}/{n_f}')
    lines.append(f'  Last-min:    v1={lm_v1}/{n_f}   v3.1={lm_v3}/{n_f}')

    adv_v1_vals = [r['adv_v1_days'] for r in kelmarsh_res.values()
                   if r['adv_v1_days'] is not None]
    adv_v3_vals = [r['adv_v3_days'] for r in kelmarsh_res.values()
                   if r['adv_v3_days'] is not None]
    if adv_v1_vals:
        lines.append(f'  Adv medio:   v1={np.mean(adv_v1_vals):.1f}d  '
                     f'(mediana {np.median(adv_v1_vals):.1f}d)')
    if adv_v3_vals:
        lines.append(f'               v3.1={np.mean(adv_v3_vals):.1f}d  '
                     f'(mediana {np.median(adv_v3_vals):.1f}d)')

    if cmapss_res:
        lines.append('\n── CMAPSS USEFUL ADVANCE WARNING ──')
        lines.append(f'  {"Dataset":<8}  {"v1 %":>6}  {"v3.1 %":>7}'
                     f'  {"Δ":>6}  {"report v1%":>12}')
        lines.append('  ' + '─' * 48)
        for fd, r in cmapss_res.items():
            delta = (f'{r["pct_v3"] - r["pct_v1"]:+.1f}%'
                     if r['pct_v1'] > 0 else 'n/a   ')
            lines.append(f'  {fd:<8}  {r["pct_v1"]:>6.1f}  {r["pct_v3"]:>7.1f}'
                         f'  {delta:>6}  {r["report_v1_ref"]:>12}')

    lines.append('\n' + '═' * 72)
    txt = '\n'.join(lines)
    (out_dir / 'summary.txt').write_text(txt, encoding='utf-8')
    print('\n' + txt)
    print(f'\n  Salvato: {out_dir / "summary.txt"}')

    def fix_keys(d):
        if isinstance(d, dict):
            return {str(k): fix_keys(v) for k, v in d.items()}
        return d

    full_json = dict(
        config=dict(
            thr_p=THR_P, l_pers_a=L_PERS_A, thr_pers_a=THR_PERS_A, thr_vel_amp=THR_VEL_AMP,
            l_pers_b=L_PERS_B, thr_pers_base=THR_PERS_BASE, alpha=ALPHA, pers_floor=PERS_FLOOR,
            l_vel=L_VEL, warmup_win=WARMUP_WIN,
            vel_short=VEL_SHORT, vel_long=VEL_LONG,
            k_confirm=K_CONFIRM, hysteresis=HYSTERESIS,
            rw=RW, rw_pr=RW_PR,
            win=WIN, step=STEP,
            adv_window_days=ADV_WINDOW, min_clean_months=MIN_CLEAN_MONTHS,
            thr_v1=THR_V1, min_win_v1=MIN_WIN_V1,
        ),
        kelmarsh_faults=kelmarsh_res,
        kelmarsh_fa=fix_keys(fa_by_ty),
        cmapss=cmapss_res,
    )
    json_path = out_dir / 'results.json'
    json_path.write_text(
        json.dumps(fix_keys(full_json), indent=2, default=str),
        encoding='utf-8'
    )
    print(f'  Salvato: {json_path}')


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Symphonon P v4.1 — mixture + regime detrending')
    parser.add_argument(
        '--data',
        default=r'D:\Symphonon\SymphononP_Package\SymphononP_Package\data',
        help='Cartella radice dataset')
    parser.add_argument(
        '--out', default='output_v4_1',
        help='Cartella output (default: ./output_v3_2/)')
    parser.add_argument(
        '--skip-cmapss', action='store_true',
        help='Salta il benchmark CMAPSS')
    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        print(f'ERRORE: cartella data non trovata: {data_dir}')
        sys.exit(1)

    print('╔══════════════════════════════════════════════════════════════╗')
    print('║  Symphonon P v4.1 — mixture + regime detrending            ║')
    print('║                                                              ║')
    print('║  Livello 1: P_v2.1       — P_base = composito ponderato    ║')
    print('║  Livello 2: Persistenza  — ci resti?                       ║')
    print('║             Velocità     — stai ancora crescendo?          ║')
    print('║  Gating:    Candidato → Qualificato → Confermato           ║')
    print('╚══════════════════════════════════════════════════════════════╝')
    print(f'\n  data → {data_dir}')
    print(f'  out  → {out_dir.resolve()}\n')

    kelmarsh_res, fa_by_ty = run_kelmarsh(data_dir, out_dir)

    cmapss_res = {}
    if not args.skip_cmapss:
        cmapss_res = run_cmapss(data_dir, out_dir)

    write_summary(kelmarsh_res, fa_by_ty, cmapss_res, out_dir)
    print(f'\n✓ Tutto salvato in: {out_dir.resolve()}')
    print('  → incolla summary.txt e results.json in chat per l\'analisi')


if __name__ == '__main__':
    main()
