"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  BOOTSTRAP CI — Kelmarsh Kelmarsh validazione Symphonon P                   ║
║                                                                              ║
║  Problema: con soli 7 fault events, le metriche aggregate sono fragili.     ║
║  Soluzione: bootstrap stratificato (fault events + clean turbine-years)     ║
║                                                                              ║
║  Procedura:                                                                  ║
║    1. Esegue P su tutti 36 turbine-year (6T × 6A)                           ║
║    2. Separa in: fault_events (7) e clean_ty (29)                           ║
║    3. Bootstrap N_BOOT=2000 iterazioni:                                      ║
║       - Ricampiona con rimpiazzo i fault_events                              ║
║       - Ricampiona con rimpiazzo i clean_ty                                  ║
║       - Calcola: detection_rate, mean_advance, FA/month                     ║
║    4. Report: media ± std, CI 95% (percentile)                              ║
║    5. Sensibilità al labeling: escludi T3-2018 come "plausible" vs         ║
║       "missed" e mostra quanto cambia il detection rate                     ║
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
from wind_turbine_precursor import extract_sensors, normalize_and_correct
from wind_ablation import (
    load_turbine, compute_components, build_signal, measure_advance, fa_rate
)

THR      = 0.80
N_BOOT   = 2000
ADV_WINDOW = 60
np.random.seed(42)

FAULTS = {
    (3, 2016): ('Oscillation encoder',  '2016-03-01 18:02', False),  # (name, dt, is_acute)
    (1, 2017): ('High rotor speed',     '2017-01-18 23:19', False),
    (6, 2017): ('Freq converter T6',    '2017-11-11 20:05', False),
    (3, 2018): ('Freq converter T3',    '2018-04-20 05:31', False),  # disputed
    (1, 2020): ('Low hydraulic press',  '2020-03-25 10:57', True),   # acute
    (3, 2020): ('Freq converter T3b',   '2020-10-24 12:40', False),
    (4, 2020): ('Freq converter T4',    '2020-08-21 08:50', False),
}

TURBINES = [1, 2, 3, 4, 5, 6]
YEARS    = [2016, 2017, 2018, 2019, 2020, 2021]
ZIPS     = {y: f'data/wind_turbine/Kelmarsh_{y}.zip' for y in YEARS}

WEIGHTS_FULL = dict(wn=0.45, wk=0.30, wc=0.25)


def main():
    print("Bootstrap CI — Symphonon P on Kelmarsh 6T × 6Y")
    print("=" * 60)

    # ── 1. Load all turbine-years ─────────────────────────────────────────────
    print("\n[1/4] Loading all turbine-years...")
    ty_data = {}

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
                ty_data[(tid, year)] = (t_dt, n_rs, k_rs, c_rs)
                print(f"   T{tid} {year}: {len(t_dt)} windows")

    print(f"\n   Total turbine-years loaded: {len(ty_data)}")

    # ── 2. Point estimates ────────────────────────────────────────────────────
    print("\n[2/4] Point estimates at θ = 0.80...")
    wn, wk, wc = WEIGHTS_FULL['wn'], WEIGHTS_FULL['wk'], WEIGHTS_FULL['wc']

    fault_results  = {}   # (tid,year) -> {advance, detected}
    clean_fa_rates = []   # FA/month for clean ty

    for key, (t_dt, n_rs, k_rs, c_rs) in ty_data.items():
        tid, year = key
        sig = build_signal(n_rs, k_rs, c_rs, wn, wk, wc)

        if key in FAULTS:
            fname, fdt, is_acute = FAULTS[key]
            adv = measure_advance(t_dt, sig, fdt, THR)
            fault_results[key] = {
                'name': fname, 'dt': fdt, 'is_acute': is_acute,
                'advance': adv,
                'detected': adv is not None and adv > 0
            }
            print(f"   T{tid} {year} [{fname}]: "
                  f"{'DETECTED +' + str(round(adv,1)) + 'd' if adv and adv>0 else 'MISSED'}"
                  f"{'  (acute)' if is_acute else ''}")
        else:
            fa = fa_rate(t_dt, sig, None, THR, year)
            clean_fa_rates.append(fa)

    n_detected = sum(1 for r in fault_results.values() if r['detected'])
    n_total    = len(fault_results)
    mean_adv   = np.mean([r['advance'] for r in fault_results.values()
                          if r['detected'] and r['advance'] is not None])
    mean_fa    = np.mean(clean_fa_rates) if clean_fa_rates else 0

    print(f"\n   Point estimate: {n_detected}/{n_total} detected  "
          f"({100*n_detected/n_total:.1f}%)")
    print(f"   Mean advance:   {mean_adv:.1f} days")
    print(f"   FA/month:       {mean_fa:.3f}")

    # ── 3. Bootstrap ──────────────────────────────────────────────────────────
    print(f"\n[3/4] Bootstrap (N={N_BOOT} iterations)...")

    fault_keys  = list(fault_results.keys())
    clean_fas   = np.array(clean_fa_rates)

    boot_det    = []
    boot_adv    = []
    boot_fa     = []

    for _ in range(N_BOOT):
        # Resample fault events
        idx_f = np.random.choice(len(fault_keys), size=len(fault_keys), replace=True)
        boot_fkeys = [fault_keys[i] for i in idx_f]

        det_count = 0
        advances_b = []
        for k in boot_fkeys:
            r = fault_results[k]
            if r['detected']:
                det_count += 1
                advances_b.append(r['advance'])

        boot_det.append(det_count / len(fault_keys))
        boot_adv.append(float(np.mean(advances_b)) if advances_b else 0.0)

        # Resample clean FA rates
        idx_c = np.random.choice(len(clean_fas), size=len(clean_fas), replace=True)
        boot_fa.append(float(clean_fas[idx_c].mean()))

    boot_det  = np.array(boot_det)
    boot_adv  = np.array(boot_adv)
    boot_fa   = np.array(boot_fa)

    ci_det = np.percentile(boot_det, [2.5, 97.5])
    ci_adv = np.percentile(boot_adv, [2.5, 97.5])
    ci_fa  = np.percentile(boot_fa,  [2.5, 97.5])

    print(f"\n   Bootstrap 95% CI:")
    print(f"   Detection rate:   {boot_det.mean():.3f}  CI [{ci_det[0]:.3f}, {ci_det[1]:.3f}]")
    print(f"   Mean advance:     {boot_adv.mean():.1f}d  CI [{ci_adv[0]:.1f}, {ci_adv[1]:.1f}]")
    print(f"   FA/month:         {boot_fa.mean():.3f}  CI [{ci_fa[0]:.3f}, {ci_fa[1]:.3f}]")

    # ── 4. Labeling sensitivity ───────────────────────────────────────────────
    print("\n[3b/4] Labeling sensitivity (T3-2018 disputed)...")

    # Scenario A: T3-2018 as TRUE POSITIVE (default)
    det_A = n_detected
    # Scenario B: T3-2018 as MISSED
    det_B = sum(1 for k, r in fault_results.items()
                if r['detected'] and k != (3, 2018))
    # Scenario C: T1-2020 (acute) excluded from count
    det_C = sum(1 for k, r in fault_results.items()
                if r['detected'] and k != (1, 2020))
    # Scenario D: both T3-2018 disputed + T1-2020 excluded
    det_D = sum(1 for k, r in fault_results.items()
                if r['detected'] and k not in [(3, 2018), (1, 2020)])

    print(f"\n   Labeling scenarios (out of {n_total} documented faults):")
    print(f"   A (default): T3-2018 as TP     → {det_A}/{n_total} ({100*det_A/n_total:.1f}%)")
    print(f"   B: T3-2018 as missed            → {det_B}/{n_total} ({100*det_B/n_total:.1f}%)")
    print(f"   C: T1-2020 excluded (acute)     → {det_C}/{n_total} ({100*det_C/n_total:.1f}%)")
    print(f"   D: T3-2018 missed + T1 excluded → {det_D}/{n_total} ({100*det_D/n_total:.1f}%)")

    # ── 5. Plots ──────────────────────────────────────────────────────────────
    print("\n[4/4] Building plots...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Bootstrap Analysis — Symphonon P on Kelmarsh 2016–2021\n'
                 f'N={N_BOOT} bootstrap iterations, θ = {THR}', fontsize=11)

    titles  = ['Detection Rate', 'Mean Advance Warning (days)', 'FA / month']
    datas   = [boot_det * 100, boot_adv, boot_fa]
    units   = ['%', 'days', 'FA/month']
    pts     = [n_detected/n_total*100, mean_adv, mean_fa]
    ci_list = [ci_det*100, ci_adv, ci_fa]

    for ax, data, title, unit, pt, ci in zip(axes, datas, titles, units, pts, ci_list):
        ax.hist(data, bins=40, color='#4488cc', alpha=0.7, edgecolor='white', lw=0.3)
        ax.axvline(pt,    color='#ff4444', lw=2, label=f'Point est.: {pt:.1f} {unit}')
        ax.axvline(ci[0], color='#ffaa44', lw=1.5, ls='--',
                   label=f'95% CI: [{ci[0]:.1f}, {ci[1]:.1f}]')
        ax.axvline(ci[1], color='#ffaa44', lw=1.5, ls='--')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel(unit)
        ax.set_ylabel('Bootstrap count')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('wind_bootstrap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   Saved: wind_bootstrap.png")

    # Labeling sensitivity bar chart
    fig2, ax = plt.subplots(figsize=(8, 4))
    scenarios = ['A\n(default,\nT3-2018=TP)',
                 'B\n(T3-2018\nmissed)',
                 'C\n(T1-2020\nexcluded)',
                 'D\n(T3-2018 missed\n+T1 excluded)']
    det_pcts = [det_A/n_total*100, det_B/n_total*100,
                det_C/n_total*100, det_D/n_total*100]
    cols = ['#44cc44' if d >= 80 else '#ffaa44' if d >= 60 else '#ff4444'
            for d in det_pcts]
    bars = ax.bar(range(4), det_pcts, color=cols, alpha=0.85, edgecolor='white')
    ax.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=10)
    ax.set_xticks(range(4))
    ax.set_xticklabels(scenarios, fontsize=9)
    ax.set_ylabel('Detection Rate (%)')
    ax.set_ylim(0, 115)
    ax.set_title('Sensitivity to Fault Labeling Policy', fontweight='bold')
    ax.axhline(80, color='#44cc44', ls='--', lw=1, alpha=0.6, label='80% target')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('wind_bootstrap_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   Saved: wind_bootstrap_sensitivity.png")

    # ── Save results ──────────────────────────────────────────────────────────
    output = {
        'n_fault_events': n_total,
        'n_detected_point': int(n_detected),
        'detection_rate_point': round(n_detected / n_total, 3),
        'mean_advance_days_point': round(float(mean_adv), 1),
        'fa_per_month_point': round(float(mean_fa), 3),
        'bootstrap_n': N_BOOT,
        'threshold': THR,
        'bootstrap': {
            'detection_rate': {
                'mean': round(float(boot_det.mean()), 3),
                'std':  round(float(boot_det.std()), 3),
                'ci95': [round(float(ci_det[0]), 3), round(float(ci_det[1]), 3)]
            },
            'mean_advance_days': {
                'mean': round(float(boot_adv.mean()), 1),
                'std':  round(float(boot_adv.std()), 1),
                'ci95': [round(float(ci_adv[0]), 1), round(float(ci_adv[1]), 1)]
            },
            'fa_per_month': {
                'mean': round(float(boot_fa.mean()), 3),
                'std':  round(float(boot_fa.std()), 3),
                'ci95': [round(float(ci_fa[0]), 3), round(float(ci_fa[1]), 3)]
            }
        },
        'labeling_sensitivity': {
            'A_T3_2018_as_TP':   {'detected': det_A, 'rate': round(det_A/n_total, 3)},
            'B_T3_2018_missed':  {'detected': det_B, 'rate': round(det_B/n_total, 3)},
            'C_T1_2020_excl':    {'detected': det_C, 'rate': round(det_C/n_total, 3)},
            'D_T3_missed_T1_ex': {'detected': det_D, 'rate': round(det_D/n_total, 3)},
        },
        'fault_details': {
            str(k): {
                'name': v['name'],
                'advance_days': round(v['advance'], 1) if v['advance'] else None,
                'detected': bool(v['detected']),
                'is_acute': bool(v['is_acute'])
            }
            for k, v in fault_results.items()
        }
    }

    with open('wind_bootstrap_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("   Saved: wind_bootstrap_results.json")

    return output


if __name__ == '__main__':
    main()
