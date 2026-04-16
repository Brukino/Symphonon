"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  SYMPHONON — Precursor su turbine eoliche reali                             ║
║                                                                              ║
║  Dataset: Kelmarsh Wind Farm (UK) — 6 turbine Senvion MM92 — 2016          ║
║  Dati:    SCADA 10 minuti, ~60 segnali per turbina                          ║
║  Fonte:   Zenodo 10.5281/zenodo.5841834  (CC-BY-4.0)                        ║
║                                                                              ║
║  Sensori usati: temperature componenti (col 90-104)                         ║
║    → dovrebbero essere indipendenti in regime sano                          ║
║    → si correlano quando un componente inizia a degradare                   ║
║    → compress (PC1 ratio) cattura questa perdita di indipendenza            ║
║                                                                              ║
║  Correzione operativa:                                                       ║
║    Il vento cambia continuamente → 3 regimi di potenza                      ║
║    Idle (<200kW), Partial (200-1500kW), Rated (>1500kW)                     ║
║    → sottrae la media di regime per residui puri di salute                  ║
║                                                                              ║
║  Guasti target (meccanici, non grid-events):                                ║
║    T1: Emergency stop nacelle [111]  2016-01-14  (211h fermo)               ║
║    T5: Low gearbox oil pressure [1510] 2016-01-28 (95h fermo)               ║
║    T1: Gear oil pump overload [1800]  2016-05-26  (9h fermo)                ║
║    T1: Feedback brake [2100]          2016-04-25  (24h fermo)               ║
║    T3: Oscillation encoder tower[4588]2016-03-01  (16h fermo)               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import pandas as pd
import zipfile
from pathlib import Path
import sys, warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("data/wind_turbine")
ZIP_2016 = DATA_DIR / "Kelmarsh_2016.zip"

# ─── Finestre ────────────────────────────────────────────────────────────────
WIN  = 144   # 24h di dati a 10 min
STEP = 24    # step 4h

# ─── Colonne sensori (identificate dall'analisi esplorativa) ─────────────────
# Temperature componenti: col 90-104 → range -20..120°C
TEMP_COLS = list(range(90, 105))
# RPM generatore
RPM_COLS  = [182, 183, 184, 185]
# Vibrazione / oscillazione (identificate empiricamente):
#   col 241: Tower acceleration X  → precursore forte per T3 (27x aumento pre-fault)
#   col 243: Torre / componente secondario
#   col 273: Drive train oscillation (alta variabilità, cattura regime meccanico)
VIB_COLS  = [241, 243, 273]
# Potenza attiva (per correzione operativa)
POWER_COL = 62
# Wind speed (per correlazione/reference)
WIND_COL  = 1

# ─── Guasti meccanici di interesse ───────────────────────────────────────────
FAULTS = {
    1: [
        ('Emergency stop nacelle',    '2016-01-14 19:28', 211.0, '#ff4444'),
        ('Freq. converter error',     '2016-01-24 16:51', 195.0, '#ff8844'),
        ('Feedback brake',            '2016-04-25 16:24',  24.0, '#ffaa44'),
        ('Gear oil pump overload',    '2016-05-26 08:34',   9.4, '#ffdd44'),
        ('Feedback brake (2)',        '2016-06-09 17:48',  18.9, '#ffaa44'),
    ],
    5: [
        ('Low gearbox oil pressure',  '2016-01-28 16:01',  95.0, '#44aaff'),
    ],
    3: [
        ('Oscillation encoder tower', '2016-03-01 18:02',  16.6, '#44ffaa'),
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# CARICA DATI
# ─────────────────────────────────────────────────────────────────────────────

def load_turbine(turbine_id, zippath=ZIP_2016):
    with zipfile.ZipFile(zippath) as z:
        fname = [f for f in z.namelist()
                 if f'Turbine_Data_Kelmarsh_{turbine_id}_' in f][0]
        with z.open(fname) as f:
            df = pd.read_csv(f, comment='#', header=None,
                             index_col=0, parse_dates=True)
    df.sort_index(inplace=True)
    return df


def extract_sensors(df):
    """
    Estrae: temperature (TEMP_COLS), RPM (RPM_COLS), power (POWER_COL).
    Forward-fill piccoli gap (≤6 step = 1h), poi dropna.
    """
    cols_want = TEMP_COLS + RPM_COLS + VIB_COLS + [POWER_COL, WIND_COL]
    available = [c for c in cols_want if c in df.columns]
    sub = df[available].copy()

    # Forward-fill gap piccoli (≤6 step = 60 min)
    sub = sub.ffill(limit=6)

    # Conta dati disponibili
    coverage = sub.notna().mean()
    good_cols = coverage[coverage > 0.45].index.tolist()  # reale SCADA: 65% è ottimo

    # Separa sensori salute vs. reference
    health_cols = [c for c in good_cols
                   if c in TEMP_COLS + RPM_COLS + VIB_COLS]
    power_series = sub[POWER_COL] if POWER_COL in sub.columns else None
    wind_series  = sub[WIND_COL]  if WIND_COL  in sub.columns else None

    sensors_df = sub[health_cols].dropna(how='all')
    print(f"   sensori salute: {len(health_cols)}  "
          f"(temp={sum(c in TEMP_COLS for c in health_cols)}, "
          f"RPM={sum(c in RPM_COLS for c in health_cols)}, "
          f"vib={sum(c in VIB_COLS for c in health_cols)})")

    return sensors_df, power_series, wind_series


# ─────────────────────────────────────────────────────────────────────────────
# CORREZIONE OPERATIVA
# ─────────────────────────────────────────────────────────────────────────────

def power_regime(power_val):
    """3 regimi: 0=idle, 1=partial, 2=rated"""
    if pd.isna(power_val): return -1
    if power_val < 200:    return 0
    if power_val < 1500:   return 1
    return 2


def normalize_and_correct(sensors_df, power_series):
    """
    Z-score globale + sottrai la media di regime operativo.
    Ritorna array numpy (T, N_sensors).
    """
    # Allinea power su stessa timeline
    if power_series is not None:
        p_aligned = power_series.reindex(sensors_df.index).ffill()
    else:
        p_aligned = pd.Series(np.nan, index=sensors_df.index)

    arr  = sensors_df.values.astype(np.float64)
    T, N = arr.shape

    # Imputa NaN con media colonna
    col_means = np.nanmean(arr, axis=0)
    for j in range(N):
        mask = np.isnan(arr[:, j])
        arr[mask, j] = col_means[j]

    # Z-score globale
    mu  = arr.mean(axis=0)
    std = arr.std(axis=0) + 1e-8
    arr_z = (arr - mu) / std

    # Correzione per regime di potenza
    regimes = np.array([power_regime(p) for p in p_aligned.values])
    arr_corr = arr_z.copy()
    for r in [0, 1, 2]:
        mask = regimes == r
        if mask.sum() > 10:
            arr_corr[mask] -= arr_z[mask].mean(axis=0)

    return arr_corr, sensors_df.index


# ─────────────────────────────────────────────────────────────────────────────
# SEGNALE SYMPHONON P
# ─────────────────────────────────────────────────────────────────────────────

def compute_P(sensors, dates, win=WIN, step=STEP):
    T, N = sensors.shape
    t_idx, k_arr, n_arr, c_arr = [], [], [], []

    baseline = sensors[:max(1, win * 3)].mean(axis=0)  # baseline = prime 3 finestre

    for s in range(0, T - win, step):
        W = sensors[s:s+win]

        # Skip finestre con troppi NaN
        if np.isnan(W).mean() > 0.2:
            continue

        t_idx.append(s + win // 2)

        # kap
        ts  = np.nanstd(W, axis=0)
        kap = float(np.clip(1.0 - np.nanmean(ts)/(np.nanstd(ts)+1e-8), 0, 1))

        # noise
        noise = float(np.clip(np.nanmean(np.abs(np.nanmean(W,axis=0)-baseline)),0,3)/3)

        # compress — PC1 ratio covarianza
        W_clean = W[~np.isnan(W).any(axis=1)]
        if W_clean.shape[0] >= 4 and N >= 2:
            eigs = np.abs(np.linalg.eigvalsh(np.cov(W_clean.T)))
            tot  = eigs.sum()
            comp = float(eigs[-1]/tot) if tot > 1e-10 else 0.0
        else:
            comp = 0.0

        k_arr.append(kap); n_arr.append(noise); c_arr.append(comp)

    # Rolling normalization (come per i mercati finanziari):
    # relativizza ogni valore alla storia recente → evita saturazione precoce
    rw = max(10, 90 // step)   # ~90 finestre = ~15 giorni di lookback

    def rolling_sigmoid(a, w):
        out = np.zeros_like(a, dtype=float)
        for i in range(len(a)):
            lo = max(0, i - w)
            mu = a[lo:i+1].mean()
            sg = a[lo:i+1].std() + 1e-8
            z  = (a[i] - mu) / sg
            out[i] = 1.0 / (1.0 + np.exp(-z))
        return out

    t    = np.array(t_idx)
    c_rs = rolling_sigmoid(np.array(c_arr), rw)
    k_rs = rolling_sigmoid(np.array(k_arr), rw)
    n_rs = rolling_sigmoid(np.array(n_arr), rw)

    P    = 0.45*n_rs + 0.30*k_rs + 0.25*c_rs
    AR   = c_rs
    t_dt = dates[t.clip(0, len(dates)-1)]
    return t_dt, P, AR, n_rs, k_rs, c_rs

    return t_dt, P, AR


# ─────────────────────────────────────────────────────────────────────────────
# ADVANCE WARNING
# ─────────────────────────────────────────────────────────────────────────────

def measure_advance(t_dt, sig, fault_dt, thr=0.70, max_days=60):
    fault_ts = pd.Timestamp(fault_dt)
    start_ts = fault_ts - pd.Timedelta(days=max_days)
    mask = (t_dt >= start_ts) & (t_dt <= fault_ts)
    if not mask.any():
        return None, None
    sub_t, sub_s = t_dt[mask], sig[mask]
    for dt, v in zip(sub_t, sub_s):
        if v >= thr:
            adv_days = (fault_ts - dt).total_seconds() / 86400
            return round(adv_days, 1), dt
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_turbine(turbine_id, t_dt, P, AR, noise_c, kap_c, comp_c,
                 sensors_df, wind_series, faults, out_dir):
    fig, axes = plt.subplots(3, 1, figsize=(16, 10),
                             gridspec_kw={'height_ratios': [1.5, 1, 1]})
    fig.patch.set_facecolor('#0a0a0a')

    for ax in axes:
        ax.set_facecolor('#111')
        ax.tick_params(colors='#aaa', labelsize=7)
        for sp in ax.spines.values(): sp.set_color('#333')

    ax1, ax2, ax3 = axes

    # ── Temperature medie (raw) ──
    temp_cols_avail = [c for c in TEMP_COLS if c in sensors_df.columns]
    if temp_cols_avail:
        temp_mean = sensors_df[temp_cols_avail].mean(axis=1).rolling(144).mean()
        ax1.plot(temp_mean.index, temp_mean.values, color='#dd8844', lw=0.8,
                 alpha=0.9, label='Temp. media (rolling 24h)')
    if wind_series is not None:
        ws = wind_series.reindex(sensors_df.index).ffill()
        ax1b = ax1.twinx()
        ax1b.plot(ws.index, ws.values, color='#4488dd', lw=0.6, alpha=0.5,
                  label='Wind speed')
        ax1b.set_ylabel('Wind (m/s)', color='#4488dd', fontsize=7)
        ax1b.tick_params(colors='#4488dd', labelsize=6)
    ax1.set_ylabel('Temp. (°C)', color='#dd8844', fontsize=8)
    ax1.set_title(f'Turbina {turbine_id} — Kelmarsh 2016',
                  color='#ddd', fontsize=10, pad=6)
    ax1.legend(fontsize=7, labelcolor='#ccc', framealpha=0.2, loc='upper left')

    # ── Segnali P / AR ──
    ax2.plot(t_dt, P,  color='#edbd38', lw=1.4, label='Symphonon P')
    ax2.plot(t_dt, AR, color='#aaaaff', lw=0.9, ls='--', alpha=0.8,
             label='AR (compress puro)')
    ax2.axhline(0.70, color='#edbd38', lw=0.7, ls=':', alpha=0.5,
                label='Soglia 0.70')
    ax2.set_ylim(-0.05, 1.10)
    ax2.set_ylabel('P / AR', color='#888', fontsize=8)
    ax2.legend(fontsize=7, labelcolor='#ccc', framealpha=0.2, loc='upper left')

    # ── Componenti separate di P ──
    ax3.plot(t_dt, noise_c, color='#ff8844', lw=0.8, alpha=0.8, label='noise (0.45)')
    ax3.plot(t_dt, kap_c,   color='#88ff44', lw=0.8, alpha=0.8, label='kap (0.30)')
    ax3.plot(t_dt, comp_c,  color='#44aaff', lw=0.8, alpha=0.8, label='compress (0.25)')
    ax3.set_ylim(-0.05, 1.10)
    ax3.set_ylabel('Componenti P', color='#888', fontsize=8)
    ax3.legend(fontsize=7, labelcolor='#ccc', framealpha=0.2, loc='upper left')

    # ── Overlay guasti ──
    for ax in (ax1, ax2, ax3):
        for fname_f, fdt, dur_h, col in faults:
            ft = pd.Timestamp(fdt)
            ax.axvline(ft, color=col, lw=1.2, ls='--', alpha=0.9)
            # zona fermo
            ax.axvspan(ft, ft + pd.Timedelta(hours=min(dur_h, 168)),
                       color=col, alpha=0.08)

    # Etichette guasti su ax2
    ypos = 1.05
    for fname_f, fdt, dur_h, col in faults:
        ft = pd.Timestamp(fdt)
        ax2.text(ft, ypos, fname_f, color=col, fontsize=5.5,
                 rotation=45, ha='left', va='bottom')

    plt.tight_layout()
    out = f'{out_dir}/wind_T{turbine_id}_2016.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
    plt.close()
    print(f"   Salvato: {out}")


def plot_zoom_fault(turbine_id, t_dt, P, AR, noise_c, kap_c, comp_c,
                    fault_name, fault_dt, days_before, col, out_dir, idx):
    """Zoom 60 giorni attorno al guasto."""
    ft     = pd.Timestamp(fault_dt)
    start  = ft - pd.Timedelta(days=days_before)
    mask   = (t_dt >= start) & (t_dt <= ft + pd.Timedelta(days=5))

    if mask.sum() < 3:
        return

    td = t_dt[mask]; Pd = P[mask]; ARd = AR[mask]

    nrd = noise_c[mask]; krd = kap_c[mask]; crd = comp_c[mask]

    fig, (ax, ax_comp) = plt.subplots(2, 1, figsize=(11, 6),
                                       gridspec_kw={'height_ratios': [2, 1]})
    fig.patch.set_facecolor('#0a0a0a')
    for ax_ in (ax, ax_comp):
        ax_.set_facecolor('#111')
        ax_.tick_params(colors='#aaa', labelsize=8)
        for sp in ax_.spines.values(): sp.set_color('#333')

    ax.plot(td, Pd,  color='#edbd38', lw=1.5, label='Symphonon P')
    ax.plot(td, ARd, color='#aaaaff', lw=1.0, ls='--', alpha=0.8, label='AR (compress)')

    # Componenti
    ax_comp.plot(td, nrd, color='#ff8844', lw=1.0, label='noise')
    ax_comp.plot(td, krd, color='#88ff44', lw=1.0, label='kap')
    ax_comp.plot(td, crd, color='#44aaff', lw=1.0, label='compress')
    ax_comp.set_ylim(-0.05, 1.10)
    ax_comp.legend(fontsize=7, labelcolor='#ccc', framealpha=0.2)
    ax_comp.set_ylabel('Componenti', color='#888', fontsize=7)
    ax.axhline(0.70, color='#edbd38', lw=0.7, ls=':', alpha=0.6)
    for ax_ in (ax, ax_comp):
        ax_.axvline(ft, color=col, lw=1.5, ls='--', alpha=0.95)
        ax_.fill_betweenx([0, 1], ft, ft + pd.Timedelta(days=5),
                          color=col, alpha=0.12)
    ax.set_ylim(-0.05, 1.10)

    # Advance warnings
    adv_P,  dt_P  = measure_advance(t_dt, P,  fault_dt)
    adv_AR, dt_AR = measure_advance(t_dt, AR, fault_dt)

    ann_text = []
    if adv_P:
        ax.axvline(dt_P, color='#edbd38', lw=1.0, ls=':')
        ann_text.append(f'P allerta: −{adv_P:.0f}gg')
    if adv_AR:
        ax.axvline(dt_AR, color='#aaaaff', lw=1.0, ls=':')
        ann_text.append(f'AR allerta: −{adv_AR:.0f}gg')
    if ann_text:
        ax.text(0.02, 0.97, '\n'.join(ann_text),
                transform=ax.transAxes, color='#ddd', fontsize=8,
                va='top', bbox=dict(facecolor='#222', alpha=0.7, pad=4))

    ax.set_title(f'T{turbine_id} — {fault_name} ({fault_dt[:10]})',
                 color='#ddd', fontsize=9)
    ax.legend(fontsize=8, labelcolor='#ccc', framealpha=0.2)

    plt.tight_layout()
    out = f'{out_dir}/wind_T{turbine_id}_fault{idx}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
    plt.close()
    print(f"   Salvato: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else '.'

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║  SYMPHONON — Turbine eoliche Kelmarsh 2016          ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    results = {}

    for turbine_id, faults in FAULTS.items():
        print(f"── Turbina {turbine_id} ──")

        # Carica
        df = load_turbine(turbine_id)
        sensors_df, power_series, wind_series = extract_sensors(df)

        # Normalizza + correggi per regime operativo
        sensors_corr, dates = normalize_and_correct(sensors_df, power_series)

        # Calcola P
        t_dt, P, AR, noise_c, kap_c, comp_c = compute_P(sensors_corr, dates)
        print(f"   {len(t_dt)} finestre computate")

        # Advance warning per ogni guasto
        print(f"   {'Guasto':35s}  {'P anticipo':12s}  {'AR anticipo':12s}  Vincitore")
        print("   " + "─"*65)
        for i, (fname_f, fdt, dur_h, col) in enumerate(faults):
            adv_P,  dt_P  = measure_advance(t_dt, P,  fdt)
            adv_AR, dt_AR = measure_advance(t_dt, AR, fdt)
            s_P  = f"+{adv_P:.1f}gg"  if adv_P  else "no alert"
            s_AR = f"+{adv_AR:.1f}gg" if adv_AR else "no alert"
            if adv_P and adv_AR:
                winner = "P" if adv_P > adv_AR else ("AR" if adv_AR > adv_P else "=")
            elif adv_P:  winner = "P solo"
            elif adv_AR: winner = "AR solo"
            else:        winner = "nessuno"
            print(f"   {fname_f:35s}  {s_P:12s}  {s_AR:12s}  {winner}")

        results[turbine_id] = (t_dt, P, AR, noise_c, kap_c, comp_c)

        # Plot timeline completa
        plot_turbine(turbine_id, t_dt, P, AR, noise_c, kap_c, comp_c,
                     sensors_df, wind_series, faults, OUT_DIR)

        # Plot zoom su ogni guasto
        for i, (fname_f, fdt, dur_h, col) in enumerate(faults):
            plot_zoom_fault(turbine_id, t_dt, P, AR, noise_c, kap_c, comp_c,
                            fname_f, fdt, 60, col, OUT_DIR, i+1)
        print()

    # ── Riepilogo ────────────────────────────────────────────────────────────
    print("═"*70)
    print("  RIEPILOGO ADVANCE WARNING  (soglia P/AR = 0.70, max 60gg)")
    print("═"*70)
    print(f"  {'Turbina':8s}  {'Guasto':35s}  {'P':10s}  {'AR':10s}  Esito")
    print("  " + "─"*66)
    for turbine_id, faults in FAULTS.items():
        t_dt, P, AR, *_ = results[turbine_id]
        for fname_f, fdt, dur_h, col in faults:
            adv_P,  _ = measure_advance(t_dt, P,  fdt)
            adv_AR, _ = measure_advance(t_dt, AR, fdt)
            s_P  = f"+{adv_P:.1f}gg"  if adv_P  else "—"
            s_AR = f"+{adv_AR:.1f}gg" if adv_AR else "—"
            if adv_P and adv_AR:
                esito = "P" if adv_P > adv_AR else "AR"
            elif adv_P:  esito = "P solo"
            elif adv_AR: esito = "AR solo"
            else:        esito = "nessuno"
            print(f"  T{turbine_id:6d}   {fname_f:35s}  {s_P:10s}  {s_AR:10s}  {esito}")
    print("═"*70)
    print("  Fatto.")
