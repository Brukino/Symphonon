"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  PENMANSHIEL VALIDATION — Symphonon P: out-of-farm replication              ║
║                                                                              ║
║  Dataset: Penmanshiel Wind Farm (UK) — 14 Senvion MM82 turbines             ║
║  Source:  Zenodo 10.5281/zenodo.5946808  (CC-BY-4.0)                        ║
║                                                                              ║
║  Obiettivo: verificare che P generalizza su una farm mai vista               ║
║    - Stessa architettura (θ=0.80, W=144, step=24, same weights)             ║
║    - Stesso tipo di turbina (Senvion MM82 vs MM92 — stessa famiglia)         ║
║    - FA rate su anni senza fault documentati                                 ║
║    - Fault detection su eventi meccanici estratti dai Status files          ║
║                                                                              ║
║  Metriche target:                                                            ║
║    FA/month ≤ 0.50  (tolleranza leggermente più alta — turbine nuove)       ║
║    Detection rate su faults meccanici estratti dai log                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import zipfile, sys, warnings, json, io
from pathlib import Path
warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from wind_ablation import (
    compute_components, build_signal, measure_advance, fa_rate, rolling_sigmoid
)

DATA_DIR = Path('data/penmanshiel')
WIN   = 144   # 24h
STEP  = 24    # 4h
THR   = 0.80
WN, WK, WC = 0.45, 0.30, 0.25

# ─── Sensor columns (by name — Penmanshiel uses named headers) ───────────────
TEMP_COLS = [
    'Front bearing temperature (°C)',
    'Rear bearing temperature (°C)',
    'Stator temperature 1 (°C)',
    'Nacelle ambient temperature (°C)',
    'Nacelle temperature (°C)',
    'Transformer temperature (°C)',
    'Gear oil inlet temperature (°C)',
    'Generator bearing rear temperature (°C)',
    'Generator bearing front temperature (°C)',
    'Gear oil temperature (°C)',
    'Rotor bearing temp (°C)',
]
RPM_COLS = [
    'Generator RPM (RPM)',
    'Rotor speed (RPM)',
    'Gearbox speed (RPM)',
]
VIB_COLS = [
    'Drive train acceleration (mm/ss)',
    'Tower Acceleration X (mm/ss)',
    'Tower Acceleration y (mm/ss)',
]
POWER_COL = 'Power (kW)'

HEALTH_COLS = TEMP_COLS + RPM_COLS + VIB_COLS

# ─── Available zip files ──────────────────────────────────────────────────────
ZIPS_WT1_10  = {
    2016: DATA_DIR / 'Penmanshiel_SCADA_2016_WT01-10_3107.zip',
    2017: DATA_DIR / 'Penmanshiel_SCADA_2017_WT01-10_3114.zip',
    2018: DATA_DIR / 'Penmanshiel_SCADA_2018_WT01-10_3113.zip',
    2019: DATA_DIR / 'Penmanshiel_SCADA_2019_WT01-10_3112.zip',
    2020: DATA_DIR / 'Penmanshiel_SCADA_2020_WT01-10_3109.zip',
    2021: DATA_DIR / 'Penmanshiel_SCADA_2021_WT01-10_3108.zip',
}
ZIPS_WT11_15 = {
    2016: DATA_DIR / 'Penmanshiel_SCADA_2016_WT11-15_3107.zip',
    2017: DATA_DIR / 'Penmanshiel_SCADA_2017_WT11-15_3115.zip',
    2018: DATA_DIR / 'Penmanshiel_SCADA_2018_WT11-15_3116.zip',
    2019: DATA_DIR / 'Penmanshiel_SCADA_2019_WT11-15_3117.zip',
    2020: DATA_DIR / 'Penmanshiel_SCADA_2020_WT11-15_3118.zip',
    2021: DATA_DIR / 'Penmanshiel_SCADA_2021_WT11-15_3108.zip',
}

# All turbine IDs (no WT03)
ALL_TID = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

# ─── Mechanical faults from literature / Status inspection ───────────────────
# These are extracted from Status files — documented Stop events with duration
# > 24h that are mechanical (not grid events, not manual stops, not tests)
# NOTE: Grid frequency stops (codes 3575/3585) and manual stops (20/21) excluded
KNOWN_FAULTS = {
    # Format: (turbine_id, 'fault_dt_str', 'description', duration_h)
    # Extracted from Status files — only prolonged mechanical stops
    # Will be populated after Status file analysis
}


def find_zip(turbine_id, year):
    """Return zip path for given turbine and year, None if not available."""
    if turbine_id <= 10:
        z = ZIPS_WT1_10.get(year)
    else:
        z = ZIPS_WT11_15.get(year)
    if z is None or not Path(z).exists():
        return None
    return z


def load_turbine(turbine_id, year):
    """Load Penmanshiel turbine SCADA data as DataFrame with named columns.

    The CSV header line starts with '# Date and time,...', so we cannot use
    comment='#' (it would skip the header). Instead we read the raw bytes,
    strip the leading '#' from the header line, then parse with pandas.
    """
    zpath = find_zip(turbine_id, year)
    if zpath is None:
        return None
    try:
        with zipfile.ZipFile(zpath) as z:
            candidates = [f for f in z.namelist()
                          if f'Turbine_Data_Penmanshiel_{turbine_id}_' in f
                          and f.endswith('.csv')]
            if not candidates:
                return None
            with z.open(candidates[0]) as f:
                raw = f.read().decode('utf-8', errors='replace')

        lines = raw.split('\n')
        # Find the header line (starts with '# Date and time')
        header_idx = next(
            (i for i, l in enumerate(lines) if 'Date and time' in l), None)
        if header_idx is None:
            return None
        # Strip leading '#' from header, keep data lines as-is
        lines[header_idx] = lines[header_idx].lstrip('#').strip()
        csv_text = '\n'.join(lines[header_idx:])
        df = pd.read_csv(io.StringIO(csv_text),
                         index_col=0, parse_dates=True, low_memory=False)
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        print(f"   ERROR loading T{turbine_id} {year}: {e}")
        return None


def load_status(turbine_id, year):
    """Load Status events file for a turbine-year."""
    zpath = find_zip(turbine_id, year)
    if zpath is None:
        return None
    try:
        with zipfile.ZipFile(zpath) as z:
            candidates = [f for f in z.namelist()
                          if f'Status_Penmanshiel_{turbine_id}_' in f
                          and f.endswith('.csv')]
            if not candidates:
                return None
            with z.open(candidates[0]) as f:
                raw = f.read().decode('utf-8', errors='replace')
        lines = raw.split('\n')
        # Find data header (first non-comment line with 'Timestamp')
        header_idx = next(i for i, l in enumerate(lines) if l.startswith('Timestamp'))
        df = pd.read_csv(io.StringIO('\n'.join(lines[header_idx:])))
        return df
    except Exception as e:
        return None


def extract_sensors_named(df):
    """Extract health sensors using named columns (Penmanshiel format)."""
    available_health = [c for c in HEALTH_COLS if c in df.columns]
    power_series = df[POWER_COL] if POWER_COL in df.columns else None

    if len(available_health) < 4:
        return None, None, None

    sensors_df = df[available_health].copy()
    sensors_df = sensors_df.ffill(limit=6)

    # Coverage filter
    coverage = sensors_df.notna().mean()
    good_cols = coverage[coverage > 0.45].index.tolist()
    if len(good_cols) < 4:
        return None, None, None

    sensors_df = sensors_df[good_cols].dropna(how='all')

    # Report sensor breakdown
    n_temp = sum(c in TEMP_COLS for c in good_cols)
    n_rpm  = sum(c in RPM_COLS for c in good_cols)
    n_vib  = sum(c in VIB_COLS for c in good_cols)
    print(f"      sensors: {len(good_cols)} (temp={n_temp}, RPM={n_rpm}, vib={n_vib})")

    return sensors_df, power_series, None


def normalize_and_correct(sensors_df, power_series):
    """Same as Kelmarsh: z-score + power regime correction."""
    if power_series is not None:
        p_aligned = power_series.reindex(sensors_df.index).ffill()
    else:
        p_aligned = pd.Series(np.nan, index=sensors_df.index)

    arr = sensors_df.values.astype(np.float64)
    T, N = arr.shape

    col_means = np.nanmean(arr, axis=0)
    for j in range(N):
        mask = np.isnan(arr[:, j])
        arr[mask, j] = col_means[j]

    mu  = arr.mean(axis=0)
    std = arr.std(axis=0) + 1e-8
    arr_z = (arr - mu) / std

    def power_regime(p):
        if pd.isna(p): return -1
        if p < 200:    return 0
        if p < 1500:   return 1
        return 2

    regimes = np.array([power_regime(p) for p in p_aligned.values])
    arr_corr = arr_z.copy()
    for r in [0, 1, 2]:
        mask = regimes == r
        if mask.sum() > 10:
            arr_corr[mask] -= arr_z[mask].mean(axis=0)

    return arr_corr, sensors_df.index


def extract_mechanical_faults(turbine_id, year):
    """
    Extract mechanical Stop events from Status file.
    Filters out: grid events (3575/3585), manual stops (20/21),
    test programs, battery tests, park master stops.
    Returns list of (fault_dt_str, description, duration_h).
    """
    GRID_CODES    = {3575, 3585, 3576, 3586, 8000}  # grid frequency, park master
    MANUAL_CODES  = {20, 21, 117, 710}               # manual stop, emergency base box, battery
    TEST_CODES    = set(range(410, 430))              # test brake programs

    df = load_status(turbine_id, year)
    if df is None:
        return []

    stops = df[df['Status'] == 'Stop'].copy()
    if stops.empty:
        return []

    faults = []
    for _, row in stops.iterrows():
        code = int(row.get('Code', 0)) if pd.notna(row.get('Code', np.nan)) else 0
        msg  = str(row.get('Message', '')).lower()

        # Skip non-mechanical
        if code in GRID_CODES or code in MANUAL_CODES or code in TEST_CODES:
            continue
        if any(kw in msg for kw in ['grid', 'manual', 'test', 'battery', 'park master',
                                     'remote stop', 'on site', 'external stop']):
            continue

        # Parse duration
        dur_str = str(row.get('Duration', '0:0:0'))
        try:
            parts = dur_str.split(':')
            dur_h = int(parts[0]) + int(parts[1]) / 60 + float(parts[2]) / 3600
        except:
            dur_h = 0

        # Only prolonged stops (> 6h) — likely mechanical
        if dur_h < 6:
            continue

        faults.append({
            'turbine': turbine_id,
            'year': year,
            'fault_dt': str(row.get('Timestamp start', '')),
            'duration_h': round(dur_h, 1),
            'code': code,
            'message': str(row.get('Message', '')),
        })

    return faults


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("PENMANSHIEL VALIDATION — Symphonon P out-of-farm replication")
    print("=" * 65)

    # ── 1. Discover available zips ────────────────────────────────────────
    available_years = sorted([y for y in ZIPS_WT11_15.keys()
                               if Path(ZIPS_WT11_15[y]).exists()])
    available_years_w1 = sorted([y for y in ZIPS_WT1_10.keys()
                                  if Path(ZIPS_WT1_10[y]).exists()])
    print(f"\nAvailable WT11-15 years: {available_years}")
    print(f"Available WT01-10 years: {available_years_w1}")

    # ── 2. Scan Status files for mechanical faults ────────────────────────
    print("\n[1/4] Scanning Status files for mechanical fault events...")
    all_faults = []
    all_tids   = []

    for year in available_years:
        for tid in [11, 12, 13, 14, 15]:
            fts = extract_mechanical_faults(tid, year)
            if fts:
                for f in fts:
                    print(f"   T{tid} {year}: +{f['duration_h']:.1f}h  [{f['code']}] {f['message']}")
                all_faults.extend(fts)
            all_tids.append((tid, year))

    for year in available_years_w1:
        for tid in [1, 2, 4, 5, 6, 7, 8, 9, 10]:
            fts = extract_mechanical_faults(tid, year)
            if fts:
                for f in fts:
                    print(f"   T{tid} {year}: +{f['duration_h']:.1f}h  [{f['code']}] {f['message']}")
                all_faults.extend(fts)
            all_tids.append((tid, year))

    print(f"\n   Total mechanical Stop events (>6h): {len(all_faults)}")

    # ── 3. Run P on each available turbine-year ───────────────────────────
    print("\n[2/4] Running Symphonon P on available turbine-years...")
    ty_results = {}

    for tid, year in all_tids:
        df = load_turbine(tid, year)
        if df is None:
            continue
        print(f"   T{tid} {year}: {len(df)} rows", end='')
        sensors_df, power_series, _ = extract_sensors_named(df)
        if sensors_df is None:
            print(" — SKIP (insufficient sensors)")
            continue

        sensors_corr, dates = normalize_and_correct(sensors_df, power_series)
        if len(dates) < WIN * 2:
            print(" — SKIP (too short)")
            continue

        t_dt, n_rs, k_rs, c_rs = compute_components(sensors_corr, dates)
        if len(t_dt) < 20:
            print(" — SKIP (too few windows)")
            continue

        sig = build_signal(n_rs, k_rs, c_rs, WN, WK, WC)

        # Check for fault events in this turbine-year
        ty_faults = [f for f in all_faults
                     if f['turbine'] == tid and f['year'] == year]

        if ty_faults:
            # Use first major fault of the year for detection check
            primary = sorted(ty_faults, key=lambda f: f['fault_dt'])[0]
            adv = measure_advance(t_dt, sig, primary['fault_dt'], THR, max_days=60)
            fa  = fa_rate(t_dt, sig, primary['fault_dt'], THR, year)
            status = f"+{adv:.1f}d" if (adv and adv > 0) else "MISSED"
        else:
            adv = None
            fa  = fa_rate(t_dt, sig, None, THR, year)
            status = "clean"

        ty_results[(tid, year)] = {
            'detected': bool(adv and adv > 0),
            'advance': adv,
            'fa': fa,
            'n_windows': len(t_dt),
            'has_fault': len(ty_faults) > 0,
            'fault_info': ty_faults[0] if ty_faults else None,
        }
        print(f"   T{tid} {year}: {status}  FA={fa:.3f}/mo")

    # ── 4. Summary ────────────────────────────────────────────────────────
    print("\n[3/4] Summary statistics")
    print("-" * 50)

    fault_tys  = {k: v for k, v in ty_results.items() if v['has_fault']}
    clean_tys  = {k: v for k, v in ty_results.items() if not v['has_fault']}

    fa_rates = [v['fa'] for v in ty_results.values()]
    fa_clean = [v['fa'] for v in clean_tys.values()]

    n_det = sum(1 for v in fault_tys.values() if v['detected'])
    n_fault = len(fault_tys)
    advances = [v['advance'] for v in fault_tys.values()
                if v['detected'] and v['advance'] is not None]

    print(f"  Turbine-years processed:  {len(ty_results)}")
    print(f"  Fault turbine-years:       {n_fault}")
    print(f"  Clean turbine-years:       {len(clean_tys)}")
    print()
    if n_fault > 0:
        print(f"  Detection rate:            {n_det}/{n_fault} ({100*n_det/max(1,n_fault):.1f}%)")
        if advances:
            print(f"  Mean advance warning:      {np.mean(advances):.1f}d")
            print(f"  Median advance warning:    {np.median(advances):.1f}d")
    print()
    print(f"  FA/month (all TY):         {np.mean(fa_rates):.3f}")
    if fa_clean:
        print(f"  FA/month (clean TY only):  {np.mean(fa_clean):.3f}")
    print()

    # Compare to Kelmarsh
    KELMARSH_FA = 0.22
    print(f"  ─── Comparison ───")
    print(f"  Kelmarsh FA/month:         {KELMARSH_FA:.3f}")
    fa_val = np.mean(fa_clean) if fa_clean else np.mean(fa_rates)
    delta  = ((fa_val - KELMARSH_FA) / KELMARSH_FA) * 100
    print(f"  Penmanshiel FA/month:      {fa_val:.3f}  ({delta:+.0f}% vs Kelmarsh)")
    replicated = fa_val < 0.50
    print(f"  Replication result:        {'PASS' if replicated else 'FAIL'} "
          f"(threshold: <0.50 FA/month)")

    # ── 5. Plots ──────────────────────────────────────────────────────────
    print("\n[4/4] Building plots...")

    if len(ty_results) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle('Symphonon P — Penmanshiel Out-of-Farm Replication\n'
                     f'θ={THR}, W={WIN}, step={STEP} (10-min samples)',
                     fontsize=11)

        # FA rate bar chart
        ax1 = axes[0]
        labels = [f"T{k[0]}\n{k[1]}" for k in ty_results.keys()]
        vals   = [v['fa'] for v in ty_results.values()]
        colors = ['#ff6666' if v['has_fault'] else '#4488cc'
                  for v in ty_results.values()]
        ax1.bar(range(len(vals)), vals, color=colors, alpha=0.8, edgecolor='white')
        ax1.axhline(KELMARSH_FA, color='#ffaa44', lw=1.5, ls='--',
                    label=f'Kelmarsh FA/mo = {KELMARSH_FA:.2f}')
        ax1.axhline(0.50, color='#44cc44', lw=1, ls=':', alpha=0.7,
                    label='Tolerance limit 0.50')
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels, fontsize=7, rotation=45)
        ax1.set_ylabel('FA / month')
        ax1.set_title('FA Rate by Turbine-Year', fontweight='bold')
        ax1.legend(fontsize=8)
        ax1.grid(axis='y', alpha=0.3)
        # Legend for colors
        from matplotlib.patches import Patch
        ax1.legend(handles=[
            Patch(color='#4488cc', label='Clean TY'),
            Patch(color='#ff6666', label='Fault TY'),
            plt.Line2D([0], [0], color='#ffaa44', ls='--', lw=1.5,
                       label=f'Kelmarsh FA={KELMARSH_FA:.2f}'),
        ], fontsize=8)

        # FA distribution
        ax2 = axes[1]
        if fa_clean:
            ax2.hist(fa_clean, bins=15, color='#4488cc', alpha=0.8,
                     edgecolor='white', label='Penmanshiel clean TY')
        ax2.axvline(KELMARSH_FA, color='#ffaa44', lw=2, ls='--',
                    label=f'Kelmarsh mean: {KELMARSH_FA:.2f}')
        if fa_clean:
            ax2.axvline(np.mean(fa_clean), color='#ff4444', lw=2,
                        label=f'Penmanshiel mean: {np.mean(fa_clean):.2f}')
        ax2.set_xlabel('FA / month')
        ax2.set_ylabel('Count')
        ax2.set_title('FA Rate Distribution (Clean TY)', fontweight='bold')
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('penmanshiel_fa_rate.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("   Saved: penmanshiel_fa_rate.png")

    # ── Save JSON results ──────────────────────────────────────────────────
    output = {
        'dataset': 'Penmanshiel Wind Farm',
        'turbine_type': 'Senvion MM82',
        'algorithm': {'theta': THR, 'WIN': WIN, 'STEP': STEP,
                      'weights': {'noise': WN, 'kap': WK, 'compress': WC}},
        'n_turbine_years': len(ty_results),
        'n_fault_ty': n_fault,
        'n_clean_ty': len(clean_tys),
        'fa_mean_all': round(float(np.mean(fa_rates)), 3) if fa_rates else None,
        'fa_mean_clean': round(float(np.mean(fa_clean)), 3) if fa_clean else None,
        'detection_rate': f"{n_det}/{n_fault}" if n_fault > 0 else 'N/A',
        'mean_advance_days': round(float(np.mean(advances)), 1) if advances else None,
        'kelmarsh_fa_for_comparison': KELMARSH_FA,
        'replication_pass': bool(replicated),
        'turbine_year_details': {
            str(k): {
                'has_fault': v['has_fault'],
                'detected': v['detected'],
                'advance_days': round(v['advance'], 1) if v['advance'] else None,
                'fa_per_month': round(v['fa'], 3),
                'n_windows': v['n_windows'],
            }
            for k, v in ty_results.items()
        },
        'mechanical_faults_found': [
            {k: v for k, v in f.items()} for f in all_faults
        ],
    }

    with open('penmanshiel_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("   Saved: penmanshiel_results.json")
    print()
    print("=" * 65)
    print("DONE")

    return output


if __name__ == '__main__':
    main()
