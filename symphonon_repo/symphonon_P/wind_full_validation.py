"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  SYMPHONON — Validazione completa 6 turbine × 6 anni (2016-2021)           ║
║                                                                              ║
║  Soglia fissa: 0.80  (calibrata su T3 2016, mai rivista)                   ║
║                                                                              ║
║  Guasti meccanici noti:                                                      ║
║    T3 2016: Oscillation encoder    [4588]  +22gg (FONTE calibrazione)       ║
║    T1 2017: High rotor speed       [805]   cluster Jan                       ║
║    T6 2017: Freq converter         [3110]  110h  Nov                        ║
║    T3 2018: Freq converter         [3110]  7h    Apr  (post-cluster Mar)    ║
║    T1 2020: Low hydraulic pressure [5510]  27h   Mar                        ║
║    T3 2020: Freq converter         [3110]  50h   Oct                        ║
║    T4 2020: Freq converter         [3110]  48h   Aug                        ║
║                                                                              ║
║  Output: heatmap FA/mese per turbina-anno + tabla advance warning           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import zipfile, sys, warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from wind_turbine_precursor import (
    extract_sensors, normalize_and_correct, compute_P
)
from wind_turbine_precursor import measure_advance
from wind_false_alarm import count_alarms, months_in_period

THR        = 0.80
YEARS      = [2016, 2017, 2018, 2019, 2020, 2021]
TURBINES   = [1, 2, 3, 4, 5, 6]

ZIPS = {y: f'data/wind_turbine/Kelmarsh_{y}.zip' for y in YEARS}

# ─── Guasti meccanici per (turbine, year) ─────────────────────────────────
# (nome, timestamp, durata_h, colore)
FAULTS = {
    (3, 2016): [('Oscillation encoder',  '2016-03-01 18:02',  16.6, '#ff4444')],
    (1, 2017): [('High rotor speed',     '2017-01-18 23:19',  11.4, '#ff8844')],
    (6, 2017): [('Freq converter T6',    '2017-11-11 20:05', 110.4, '#ffaa44')],
    (3, 2018): [('Freq converter T3',    '2018-04-20 05:31',   7.2, '#ff6688')],
    (1, 2020): [('Low hydraulic press',  '2020-03-25 10:57',  27.7, '#ff4444')],
    (3, 2020): [('Freq converter T3b',   '2020-10-24 12:40',  50.9, '#ff6688')],
    (4, 2020): [('Freq converter T4',    '2020-08-21 08:50',  48.5, '#ffaa44')],
}

# ─── Periodi da escludere per ogni (turbine, year) ────────────────────────
# (inizio, fine) — manutenzioni note, grid events prolungati, startup
BASE_EXCL = {
    2016: [('2016-01-01', '2016-02-01')],
    2017: [('2017-01-01', '2017-02-01'), ('2017-01-24', '2017-02-07')],
    2018: [('2018-01-01', '2018-02-01')],
    2019: [('2019-01-01', '2019-02-01')],
    2020: [('2020-01-01', '2020-02-01')],
    2021: [('2021-01-01', '2021-02-01')],
}


# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────

def load_year(year, turbine_id):
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


# ─────────────────────────────────────────────────────────────────────────────
# ANALISI SINGOLA (turbine, year)
# ─────────────────────────────────────────────────────────────────────────────

def analyze(turbine_id, year):
    df = load_year(year, turbine_id)
    if df is None:
        return None

    sensors_df, power_series, wind_series = extract_sensors(df)
    if sensors_df.shape[1] < 4:
        return None

    sensors_corr, dates = normalize_and_correct(sensors_df, power_series)
    t_dt, P, AR, noise_c, kap_c, comp_c = compute_P(sensors_corr, dates)

    if len(t_dt) < 50:
        return None

    # Esclusioni base
    excludes = list(BASE_EXCL.get(year, []))

    # Aggiunge esclusioni intorno a guasti noti (±35gg)
    faults = FAULTS.get((turbine_id, year), [])
    for _, fdt, dur_h, _ in faults:
        ft = pd.Timestamp(fdt)
        excludes.append((str(ft - pd.Timedelta(days=40))[:10],
                         str(ft + pd.Timedelta(days=10))[:10]))

    # Mask periodo sano
    healthy_mask = np.ones(len(t_dt), dtype=bool)
    for ex_s, ex_e in excludes:
        healthy_mask &= ~((t_dt >= pd.Timestamp(ex_s)) &
                          (t_dt <= pd.Timestamp(ex_e)))

    # Segmenti sani
    segs = []
    in_seg = False
    for i, m in enumerate(healthy_mask):
        if m and not in_seg:
            seg_s = t_dt[i]; in_seg = True
        elif not m and in_seg:
            segs.append((seg_s, t_dt[i-1])); in_seg = False
    if in_seg:
        segs.append((seg_s, t_dt[-1]))

    # FA nel periodo sano
    total_fa = 0; total_days = 0; all_eps = []
    for seg_s, seg_e in segs:
        n_fa, eps = count_alarms(t_dt, P, str(seg_s)[:19],
                                 str(seg_e)[:19], THR)
        days = (seg_e - seg_s).total_seconds() / 86400
        total_fa += n_fa; total_days += days
        all_eps.extend(eps)

    fa_month = total_fa / (total_days / 30.44) if total_days > 0 else 0.0

    # Advance warning guasti
    detections = []
    for fname_f, fdt, dur_h, col in faults:
        adv_P,  dt_P  = measure_advance(t_dt, P,  fdt, thr=THR, max_days=60)
        adv_AR, dt_AR = measure_advance(t_dt, AR, fdt, thr=THR, max_days=60)
        detections.append({
            'name': fname_f, 'fdt': fdt, 'dur_h': dur_h,
            'adv_P': adv_P, 'adv_AR': adv_AR,
            'dt_P': dt_P, 'dt_AR': dt_AR,
        })

    return {
        't_dt': t_dt, 'P': P, 'AR': AR,
        'noise_c': noise_c, 'kap_c': kap_c, 'comp_c': comp_c,
        'fa_total': total_fa, 'fa_month': fa_month,
        'healthy_days': total_days, 'episodes': all_eps,
        'detections': detections, 'segs': segs,
        'faults': faults,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else '.'

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║  SYMPHONON — 6 Turbine × 6 Anni  Kelmarsh 2016-2021║")
    print(f"║  Soglia fissa P = {THR}  (calibrata su T3 2016)        ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    # ── Esegui tutto ─────────────────────────────────────────────────────────
    results = {}
    for t_id in TURBINES:
        for year in YEARS:
            sys.stdout.write(f"  T{t_id} {year}... ")
            sys.stdout.flush()
            r = analyze(t_id, year)
            results[(t_id, year)] = r
            if r:
                det_str = ""
                for d in r['detections']:
                    s = f"+{d['adv_P']:.0f}gg" if d['adv_P'] else "—"
                    det_str += f" [{d['name'][:12]}:{s}]"
                print(f"FA={r['fa_month']:.2f}/m  {det_str}")
            else:
                print("skip")

    print()

    # ── Tabella riepilogo ─────────────────────────────────────────────────────
    print("═"*80)
    print(f"  FA/MESE per turbina-anno  (soglia={THR})")
    print("  ● = guasto rilevato   ○ = guasto non rilevato   · = nessun guasto noto")
    print("═"*80)
    header = f"  {'':6s}" + "".join(f"  {y:6d}" for y in YEARS)
    print(header)
    print("  " + "─"*70)

    fa_matrix   = np.full((len(TURBINES), len(YEARS)), np.nan)
    det_matrix  = {}  # (t_id, year) → list of detection symbols

    for t_idx, t_id in enumerate(TURBINES):
        row = f"  T{t_id}    "
        for y_idx, year in enumerate(YEARS):
            r = results.get((t_id, year))
            if r is None:
                row += "   n/a "
                continue
            fa = r['fa_month']
            fa_matrix[t_idx, y_idx] = fa

            # Simbolo guasto
            sym = " "
            for d in r['detections']:
                if d['adv_P']:
                    sym = "●"
                else:
                    sym = "○"
            if not r['detections']:
                sym = "·"

            # Colore testo in base a FA rate
            row += f"  {fa:.2f}{sym} "
        print(row)

    print("  " + "─"*70)
    # Medie per anno
    row_yr = "  Media "
    for y_idx, year in enumerate(YEARS):
        vals = fa_matrix[:, y_idx]
        vals = vals[~np.isnan(vals)]
        row_yr += f"  {vals.mean():.2f}  " if len(vals) else "   n/a "
    print(row_yr)
    print("═"*80)

    # Totale globale
    all_vals = fa_matrix[~np.isnan(fa_matrix)]
    print(f"\n  FA/mese medio globale  (6T × 6A): {all_vals.mean():.2f}")
    print(f"  FA/mese mediano globale          : {np.median(all_vals):.2f}")

    # ── Tabella advance warning ───────────────────────────────────────────────
    print()
    print("═"*75)
    print("  ADVANCE WARNING — tutti i guasti noti")
    print(f"  (soglia={THR}, finestra max 60gg)")
    print("═"*75)
    print(f"  {'Turbina':8s}  {'Anno':6s}  {'Guasto':28s}  "
          f"{'P anticipo':12s}  {'AR anticipo':12s}  Esito")
    print("  " + "─"*71)

    detected = 0; total_faults = 0
    for (t_id, year), fault_list in sorted(FAULTS.items()):
        r = results.get((t_id, year))
        if r is None:
            continue
        for d in r['detections']:
            total_faults += 1
            s_P  = f"+{d['adv_P']:.0f}gg"  if d['adv_P']  else "—"
            s_AR = f"+{d['adv_AR']:.0f}gg" if d['adv_AR'] else "—"
            if d['adv_P']:
                esito = "✓ RILEVATO"
                detected += 1
            else:
                esito = "✗ mancato"
            print(f"  T{t_id:6d}   {year}   {d['name']:28s}  "
                  f"{s_P:12s}  {s_AR:12s}  {esito}")

    print("═"*75)
    print(f"  Rilevati: {detected}/{total_faults}  "
          f"({100*detected/total_faults:.0f}%)")
    print()

    # ── HEATMAP FA/mese ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                             gridspec_kw={'width_ratios': [2.5, 1]})
    fig.patch.set_facecolor('#0a0a0a')

    # Heatmap
    ax_hm = axes[0]
    ax_hm.set_facecolor('#0a0a0a')

    # Cmap: verde (0) → giallo (0.5) → rosso (2+)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        'traffic', ['#1a4a1a', '#44cc44', '#ddcc22', '#cc4400', '#880000'])
    vmax = 2.0

    im = ax_hm.imshow(fa_matrix, aspect='auto', cmap=cmap,
                      vmin=0, vmax=vmax, interpolation='nearest')
    ax_hm.set_xticks(range(len(YEARS)))
    ax_hm.set_xticklabels([str(y) for y in YEARS], color='#aaa', fontsize=9)
    ax_hm.set_yticks(range(len(TURBINES)))
    ax_hm.set_yticklabels([f'T{t}' for t in TURBINES], color='#aaa', fontsize=9)
    ax_hm.set_title(f'False Alarms / mese  (soglia={THR})\n'
                    f'Verde=OK  Giallo=borderline  Rosso=troppi',
                    color='#ddd', fontsize=10, pad=8)

    # Testo nelle celle
    for t_idx in range(len(TURBINES)):
        for y_idx in range(len(YEARS)):
            val = fa_matrix[t_idx, y_idx]
            if np.isnan(val):
                continue
            t_id = TURBINES[t_idx]
            year = YEARS[y_idx]
            r    = results.get((t_id, year))
            sym  = ""
            if r:
                for d in r['detections']:
                    sym = "●" if d['adv_P'] else "○"
            txt_col = '#fff' if val < 0.8 else '#000'
            ax_hm.text(y_idx, t_idx, f'{val:.2f}{sym}',
                       ha='center', va='center', fontsize=8,
                       color=txt_col, fontweight='bold')

    plt.colorbar(im, ax=ax_hm, label='FA/mese', shrink=0.8)

    # Barre advance warning
    ax_adv = axes[1]
    ax_adv.set_facecolor('#111')
    ax_adv.tick_params(colors='#aaa', labelsize=8)
    for sp in ax_adv.spines.values(): sp.set_color('#333')

    det_items = []
    for (t_id, year), fault_list in sorted(FAULTS.items()):
        r = results.get((t_id, year))
        if r is None: continue
        for d in r['detections']:
            det_items.append((f"T{t_id} {year}\n{d['name'][:14]}",
                              d['adv_P'] or 0,
                              '#44cc44' if d['adv_P'] else '#cc4444'))

    if det_items:
        labels, vals, cols = zip(*det_items)
        bars = ax_adv.barh(range(len(vals)), vals, color=cols, alpha=0.85)
        ax_adv.set_yticks(range(len(labels)))
        ax_adv.set_yticklabels(labels, color='#ccc', fontsize=7)
        ax_adv.axvline(30, color='#44cc88', lw=1.0, ls='--', alpha=0.8)
        ax_adv.axvline(14, color='#ffaa44', lw=0.8, ls=':', alpha=0.7)
        ax_adv.set_xlabel('Anticipo (giorni)', color='#888', fontsize=8)
        ax_adv.set_title('Advance Warning\n● rilevato  ○ mancato',
                         color='#ddd', fontsize=9, pad=6)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax_adv.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                            f'+{val:.0f}gg', va='center', color='#ddd',
                            fontsize=7)

    plt.suptitle(f'SYMPHONON — Kelmarsh 6 Turbine × 6 Anni  |  Soglia={THR}  '
                 f'(calibrata su T3 2016 soltanto)',
                 color='#ddd', fontsize=10, y=1.01)
    plt.tight_layout()
    out = f'{OUT_DIR}/wind_full_validation.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
    plt.close()
    print(f"  Salvato: {out}")

    # ── Plot timeline T3 tutti gli anni ──────────────────────────────────────
    fig2, axes2 = plt.subplots(len(YEARS), 1, figsize=(18, 3*len(YEARS)),
                               sharex=False)
    fig2.patch.set_facecolor('#0a0a0a')
    year_cols = {2016:'#edbd38',2017:'#44aaff',2018:'#88ff88',
                 2019:'#ff88cc',2020:'#ffaa44',2021:'#88ccff'}

    for ax, year in zip(axes2, YEARS):
        r = results.get((3, year))
        col = year_cols[year]
        ax.set_facecolor('#111')
        ax.tick_params(colors='#aaa', labelsize=7)
        for sp in ax.spines.values(): sp.set_color('#333')

        if r is None:
            ax.text(0.5, 0.5, 'dati non disponibili',
                    transform=ax.transAxes, ha='center', color='#555')
            ax.set_title(f'T3 {year}', color='#666', fontsize=8)
            continue

        ax.plot(r['t_dt'], r['P'], color=col, lw=1.0, alpha=0.9)
        ax.axhline(THR, color=col, lw=0.7, ls=':', alpha=0.5)
        ax.set_ylim(-0.05, 1.20)

        # Periodi sani
        for seg_s, seg_e in r['segs']:
            ax.axvspan(seg_s, seg_e, color='#225522', alpha=0.12)

        # FA episodes
        for ep_s, dur_h in r['episodes']:
            ax.axvspan(ep_s, ep_s + pd.Timedelta(hours=max(dur_h,8)),
                       color='#ff8800', alpha=0.6, zorder=3)

        # Guasti
        for _, fdt, dur_h, fcol in r['faults']:
            ft = pd.Timestamp(fdt)
            ax.axvline(ft, color='#ff3333', lw=1.5, ls='--')
            ax.fill_betweenx([0,1], ft-pd.Timedelta(days=40), ft,
                             color='#ff3333', alpha=0.07)

        # Advance alert
        for d in r['detections']:
            if d['adv_P'] and d['dt_P']:
                ax.axvline(d['dt_P'], color=col, lw=1.2, ls=':')
                ax.text(d['dt_P'], 1.10, f"−{d['adv_P']:.0f}gg",
                        color=col, fontsize=6, ha='center')

        fa_str = f"FA={r['fa_month']:.2f}/m ({r['fa_total']} in {r['healthy_days']:.0f}gg)"
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.set_title(f'T3  {year}  —  {fa_str}',
                     color='#ddd', fontsize=8, pad=4)
        ax.set_ylabel('P', color='#888', fontsize=7)

    plt.suptitle('T3 Kelmarsh 2016-2021 — P signal  '
                 '(verde=sano, arancio=FA, rosso=guasto)',
                 color='#ddd', fontsize=10, y=1.01)
    plt.tight_layout()
    out2 = f'{OUT_DIR}/wind_T3_2016_2021.png'
    plt.savefig(out2, dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
    plt.close()
    print(f"  Salvato: {out2}")
    print("\n  Fatto.")
