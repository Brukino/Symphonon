"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ABLATION STUDY — Symphonon P componenti                                    ║
║                                                                              ║
║  Domanda: quanto contribuisce ogni componente a P?                          ║
║  Varianti testate (pesi ri-normalizzati a somma=1):                         ║
║    1. compress solo     (= AR)                                               ║
║    2. compress + noise  (no κ)                                               ║
║    3. compress + κ      (no noise)                                           ║
║    4. noise + κ         (no compress)                                        ║
║    5. P completo        (compress + noise + κ)                               ║
║                                                                              ║
║  Metrica: per ogni variante e ogni soglia θ:                                ║
║    - advance warning su fault events                                         ║
║    - FA/month su periodi senza fault                                         ║
║  Risultato: curva tradeoff per ogni variante, tabella al punto ottimale     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import zipfile, sys, warnings, json
warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from wind_turbine_precursor import (
    extract_sensors, normalize_and_correct, WIN, STEP
)
from wind_false_alarm import count_alarms, months_in_period

DATA_DIR = 'data/wind_turbine'
ZIPS = {y: f'{DATA_DIR}/Kelmarsh_{y}.zip' for y in range(2016, 2022)}

# ─── Guasti meccanici noti ────────────────────────────────────────────────────
FAULTS = {
    (3, 2016): ('Oscillation encoder',  '2016-03-01 18:02'),
    (1, 2017): ('High rotor speed',     '2017-01-18 23:19'),
    (6, 2017): ('Freq converter T6',    '2017-11-11 20:05'),
    (3, 2018): ('Freq converter T3',    '2018-04-20 05:31'),
    (1, 2020): ('Low hydraulic press',  '2020-03-25 10:57'),  # acute
    (3, 2020): ('Freq converter T3b',   '2020-10-24 12:40'),
    (4, 2020): ('Freq converter T4',    '2020-08-21 08:50'),
}
TURBINES = [1, 2, 3, 4, 5, 6]
YEARS    = [2016, 2017, 2018, 2019, 2020, 2021]

THRESHOLDS = np.arange(0.50, 0.91, 0.05)
ADV_WINDOW = 60   # giorni massimi per advance warning

# ─── Definizioni varianti ─────────────────────────────────────────────────────
VARIANTS = {
    'compress_only': dict(wn=0.0,   wk=0.0,   wc=1.0),
    'compress+noise': dict(wn=0.643, wk=0.0,   wc=0.357),  # 0.45/0.70, 0.25/0.70
    'compress+kap':  dict(wn=0.0,   wk=0.545, wc=0.455),  # 0.30/0.55, 0.25/0.55
    'noise+kap':     dict(wn=0.6,   wk=0.4,   wc=0.0),    # 0.45/0.75, 0.30/0.75
    'P_full':        dict(wn=0.45,  wk=0.30,  wc=0.25),
}

# ─── Load + compute components ───────────────────────────────────────────────

def load_turbine(turbine_id, year):
    zpath = ZIPS[year]
    with zipfile.ZipFile(zpath) as z:
        candidates = [f for f in z.namelist()
                      if f'Turbine_Data_Kelmarsh_{turbine_id}_' in f]
        if not candidates:
            return None
        with z.open(candidates[0]) as f:
            df = pd.read_csv(f, comment='#', header=None,
                             index_col=0, parse_dates=True)
    df.sort_index(inplace=True)
    return df


def rolling_sigmoid(a, w):
    out = np.zeros_like(a, dtype=float)
    for i in range(len(a)):
        lo = max(0, i - w)
        mu = a[lo:i+1].mean()
        sg = a[lo:i+1].std() + 1e-8
        z  = (a[i] - mu) / sg
        out[i] = 1.0 / (1.0 + np.exp(-z))
    return out


def compute_components(sensors, dates, win=WIN, step=STEP):
    """Returns (t_dt, noise_rs, kap_rs, comp_rs) — rolling-sigmoid normalized."""
    T, N = sensors.shape
    t_idx, k_arr, n_arr, c_arr = [], [], [], []
    baseline = sensors[:max(1, win * 3)].mean(axis=0)
    rw = max(10, 90 // step)

    for s in range(0, T - win, step):
        W = sensors[s:s+win]
        if np.isnan(W).mean() > 0.2:
            continue
        t_idx.append(s + win // 2)

        # kap
        ts  = np.nanstd(W, axis=0)
        kap = float(np.clip(1.0 - np.nanmean(ts) / (np.nanstd(ts) + 1e-8), 0, 1))

        # noise
        noise = float(np.clip(
            np.nanmean(np.abs(np.nanmean(W, axis=0) - baseline)), 0, 3) / 3)

        # compress
        W_clean = W[~np.isnan(W).any(axis=1)]
        if W_clean.shape[0] >= 4 and N >= 2:
            eigs = np.abs(np.linalg.eigvalsh(np.cov(W_clean.T)))
            tot  = eigs.sum()
            comp = float(eigs[-1] / tot) if tot > 1e-10 else 0.0
        else:
            comp = 0.0

        k_arr.append(kap); n_arr.append(noise); c_arr.append(comp)

    t     = np.array(t_idx)
    n_rs  = rolling_sigmoid(np.array(n_arr), rw)
    k_rs  = rolling_sigmoid(np.array(k_arr), rw)
    c_rs  = rolling_sigmoid(np.array(c_arr), rw)
    t_dt  = dates[t.clip(0, len(dates) - 1)]
    return t_dt, n_rs, k_rs, c_rs


def build_signal(n_rs, k_rs, c_rs, wn, wk, wc):
    return wn * n_rs + wk * k_rs + wc * c_rs


def measure_advance(t_dt, sig, fault_dt, thr, max_days=ADV_WINDOW):
    fault_ts = pd.Timestamp(fault_dt)
    start_ts = fault_ts - pd.Timedelta(days=max_days)
    mask = (t_dt >= start_ts) & (t_dt <= fault_ts)
    if not mask.any():
        return None
    for dt, v in zip(t_dt[mask], sig[mask]):
        if v >= thr:
            return (fault_ts - dt).total_seconds() / 86400
    return None

# ─── Escludi inizio anno (startup/manutenzione) ───────────────────────────────
def exclude_jan(t_dt, year):
    cutoff = pd.Timestamp(f'{year}-02-01')
    return t_dt >= cutoff

# ─── Calcola FA rate ──────────────────────────────────────────────────────────
def fa_rate(t_dt, sig, fault_dt_or_none, thr, year):
    """FA/month su periodo pulito (esclude finestra 60gg pre-fault e gen)."""
    mask_jan = exclude_jan(t_dt, year)
    sig_c = sig[mask_jan]
    t_c   = t_dt[mask_jan]

    if fault_dt_or_none is not None:
        ft = pd.Timestamp(fault_dt_or_none)
        excl_start = ft - pd.Timedelta(days=ADV_WINDOW)
        excl_end   = ft + pd.Timedelta(days=30)
        mask_ok = (t_c < excl_start) | (t_c > excl_end)
        sig_c = sig_c[mask_ok]
        t_c   = t_c[mask_ok]

    if len(t_c) < 5:
        return 0.0

    n_months = max(0.1, (t_c[-1] - t_c[0]).total_seconds() / (30.44 * 86400))

    # Hysteresis alarm counting
    alarm_on = False
    count_above = 0
    n_episodes = 0
    for v in sig_c:
        if v >= thr:
            count_above += 1
        else:
            count_above = 0
        if count_above >= 2 and not alarm_on:
            alarm_on = True
            n_episodes += 1
        if v < thr - 0.08:
            alarm_on = False
            count_above = 0

    return n_episodes / n_months


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Ablation Study — Symphonon P")
    print("=" * 60)

    # 1. Pre-compute components for all turbine-years
    print("\n[1/3] Computing components for all turbine-years...")
    components_cache = {}  # (t, y) -> (t_dt, n_rs, k_rs, c_rs)

    for year in YEARS:
        for tid in TURBINES:
            df = load_turbine(tid, year)
            if df is None:
                continue
            sensors_df, power_series, wind_series = extract_sensors(df)
            if sensors_df.shape[1] < 4:
                continue
            sensors_corr, dates = normalize_and_correct(sensors_df, power_series)
            t_dt, n_rs, k_rs, c_rs = compute_components(sensors_corr, dates)
            if len(t_dt) > 50:
                components_cache[(tid, year)] = (t_dt, n_rs, k_rs, c_rs)
                print(f"   T{tid} {year}: {len(t_dt)} windows")

    print(f"\n   Cached: {len(components_cache)} turbine-years")

    # 2. For each variant × threshold → advance warnings + FA rates
    print("\n[2/3] Evaluating variants across thresholds...")

    results = {}  # variant -> thr -> {advances, fa_rates}

    for vname, weights in VARIANTS.items():
        results[vname] = {}
        wn, wk, wc = weights['wn'], weights['wk'], weights['wc']

        for thr in THRESHOLDS:
            advances = {}   # (tid, year, fault_name) -> days or None
            fa_rates_list = []

            for (tid, year), (t_dt, n_rs, k_rs, c_rs) in components_cache.items():
                sig = build_signal(n_rs, k_rs, c_rs, wn, wk, wc)
                fault_key = (tid, year)
                fault_info = FAULTS.get(fault_key, None)

                if fault_info is not None:
                    fname, fdt = fault_info
                    adv = measure_advance(t_dt, sig, fdt, thr)
                    advances[(tid, year, fname)] = adv
                    fa = fa_rate(t_dt, sig, fdt, thr, year)
                else:
                    fa = fa_rate(t_dt, sig, None, thr, year)

                fa_rates_list.append(fa)

            results[vname][round(thr, 2)] = {
                'advances': advances,
                'fa_rates': fa_rates_list,
                'fa_mean': float(np.mean(fa_rates_list)),
                'n_detected': sum(1 for v in advances.values() if v is not None and v > 0),
                'n_faults': len(advances),
                'advances_valid': [v for v in advances.values()
                                   if v is not None and v > 0],
            }

    # 3. Summary table at optimal threshold (θ=0.80)
    print("\n[3/3] Results at θ = 0.80")
    print("\n" + "=" * 80)
    print(f"{'Variant':<20} {'Detected':>8} {'Adv (med)':>10} {'Adv (min)':>10} "
          f"{'FA/mo':>8}")
    print("-" * 80)

    summary = {}
    for vname in VARIANTS:
        r = results[vname][0.80]
        advs = r['advances_valid']
        med_adv = float(np.median(advs)) if advs else 0.0
        min_adv = float(np.min(advs)) if advs else 0.0
        det = r['n_detected']
        tot = r['n_faults']
        fa  = r['fa_mean']
        print(f"{vname:<20} {det}/{tot}{'':>3} {med_adv:>9.1f}d {min_adv:>9.1f}d "
              f"{fa:>8.2f}")
        summary[vname] = {
            'detected': det, 'total': tot,
            'median_advance_days': round(med_adv, 1),
            'min_advance_days': round(min_adv, 1),
            'fa_per_month': round(fa, 3),
            'advances': {str(k): round(v, 1) if v else None
                         for k, v in results[vname][0.80]['advances'].items()}
        }

    # 4. Tradeoff curves
    print("\nBuilding tradeoff plots...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Ablation Study — P Components (θ sweep)', fontsize=12, fontweight='bold')

    colors_v = {
        'compress_only': '#4488ff',
        'compress+noise': '#88bbff',
        'compress+kap':  '#aaffaa',
        'noise+kap':     '#ffaa44',
        'P_full':        '#ff4444',
    }
    labels_v = {
        'compress_only': 'compress only (= AR)',
        'compress+noise': 'compress + noise',
        'compress+kap':  'compress + κ',
        'noise+kap':     'noise + κ (no compress)',
        'P_full':        'P full (all 3)',
    }

    ax1, ax2 = axes

    for vname in VARIANTS:
        fa_by_thr, det_by_thr, adv_by_thr = [], [], []
        for thr in THRESHOLDS:
            r = results[vname][round(thr, 2)]
            fa_by_thr.append(r['fa_mean'])
            det_by_thr.append(r['n_detected'] / max(1, r['n_faults']))
            advs = r['advances_valid']
            adv_by_thr.append(float(np.median(advs)) if advs else 0)

        c = colors_v[vname]
        lw = 2.5 if vname == 'P_full' else 1.5
        ls = '-' if vname == 'P_full' else '--'

        ax1.plot(THRESHOLDS, fa_by_thr, color=c, lw=lw, ls=ls,
                 label=labels_v[vname])
        ax2.plot(fa_by_thr, adv_by_thr, color=c, lw=lw, ls=ls,
                 label=labels_v[vname], marker='o', ms=4)

    ax1.set_xlabel('Threshold θ'); ax1.set_ylabel('FA / month')
    ax1.set_title('False Alarm Rate vs Threshold')
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    ax1.axvline(0.80, color='gray', ls=':', lw=1)

    ax2.set_xlabel('FA / month'); ax2.set_ylabel('Median Advance Warning (days)')
    ax2.set_title('Advance Warning vs FA Rate (tradeoff)')
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('wind_ablation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   Saved: wind_ablation.png")

    # 5. Component contribution bar chart at θ=0.80
    fig2, ax = plt.subplots(figsize=(10, 5))
    vnames_ord = list(VARIANTS.keys())
    x = np.arange(len(vnames_ord))
    fa_vals  = [summary[v]['fa_per_month'] for v in vnames_ord]
    adv_vals = [summary[v]['median_advance_days'] for v in vnames_ord]
    det_vals = [summary[v]['detected'] / summary[v]['total'] * 100 for v in vnames_ord]

    ax2b = ax.twinx()
    bars1 = ax.bar(x - 0.25, fa_vals,  0.25, color='#ff6644', alpha=0.8,
                   label='FA/month (left)')
    bars2 = ax2b.bar(x,       adv_vals, 0.25, color='#4488ff', alpha=0.8,
                     label='Median advance (days, right)')
    bars3 = ax2b.bar(x + 0.25, det_vals, 0.25, color='#44cc44', alpha=0.8,
                     label='Detection rate % (right)')

    ax.set_xticks(x)
    ax.set_xticklabels([labels_v[v] for v in vnames_ord], rotation=15, ha='right')
    ax.set_ylabel('FA / month', color='#ff6644')
    ax2b.set_ylabel('Days / Detection %', color='#4488ff')
    ax.set_title('Ablation at θ = 0.80: FA Rate, Advance Warning, Detection Rate')
    ax.tick_params(axis='y', colors='#ff6644')
    ax2b.tick_params(axis='y', colors='#4488ff')

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('wind_ablation_bars.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   Saved: wind_ablation_bars.png")

    # Save JSON
    with open('wind_ablation_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("   Saved: wind_ablation_results.json")

    print("\n" + "=" * 80)
    print("KEY FINDING:")
    fa_full = summary['P_full']['fa_per_month']
    fa_ar   = summary['compress_only']['fa_per_month']
    ratio   = fa_ar / max(fa_full, 0.01)
    print(f"  compress_only FA/mo = {fa_ar:.2f}")
    print(f"  P_full        FA/mo = {fa_full:.2f}")
    print(f"  FA reduction factor = {ratio:.1f}×")
    print(f"  P_full detection    = {summary['P_full']['detected']}/{summary['P_full']['total']}")
    print(f"  compress_only det.  = {summary['compress_only']['detected']}/{summary['compress_only']['total']}")

    return summary


if __name__ == '__main__':
    from wind_turbine_precursor import extract_sensors, normalize_and_correct
    main()
