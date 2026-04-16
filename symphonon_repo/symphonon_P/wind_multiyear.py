"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  SYMPHONON — Validazione multi-anno T3 Kelmarsh (2016-2017-2018)           ║
║                                                                              ║
║  Protocollo:                                                                 ║
║    Calibrazione : 2016  (fault noto + soglia trovata = 0.80)                ║
║    Validazione  : 2017 e 2018  (out-of-sample, soglia fissa 0.80)           ║
║                                                                              ║
║  Domanda: il false alarm rate di 0.14/mese regge su anni sani?              ║
║                                                                              ║
║  Analisi parallela: stesso segnale P su tutti gli anni concatenati          ║
║  (come in produzione: il modello gira continuamente)                        ║
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
    extract_sensors, normalize_and_correct, compute_P, VIB_COLS,
    TEMP_COLS, RPM_COLS, POWER_COL, WIND_COL
)
from wind_false_alarm import count_alarms

TURBINE_ID  = 3
THR         = 0.80     # soglia calibrata su 2016
HYST        = 0.08
MIN_CONSEC  = 2
HEALTHY_EXCL_MONTHS = 1   # escludi primo mese di ogni anno (startup/calibrazione)

YEARS = [2016, 2017, 2018]
ZIPS = {
    2016: 'data/wind_turbine/Kelmarsh_2016.zip',
    2017: 'data/wind_turbine/Kelmarsh_2017.zip',
    2018: 'data/wind_turbine/Kelmarsh_2018.zip',
}

# Guasti meccanici T3 (noti da analisi precedente)
T3_FAULTS = {
    2016: [('Oscillation encoder', '2016-03-01 18:02', 16.6, '#ff4444')],
    2017: [],   # nessun guasto meccanico
    2018: [],   # freq converter (elettrico, non meccanico) — lo teniamo come "negativo"
}

# Periodi da escludere dal conteggio FA (manutenzioni note, grid events, ecc.)
T3_EXCLUDE = {
    2016: [('2016-01-01', '2016-02-01'),   # primi 30gg (calibrazione turbina)
           ('2016-02-01', '2016-03-15')],  # zona pre/post fault oscillation encoder
    2017: [('2017-01-01', '2017-02-01'),   # primo mese
           ('2017-01-24', '2017-02-05')],  # intorno al park master stop
    2018: [('2018-01-01', '2018-02-01'),   # primo mese
           ('2018-04-14', '2018-05-01')],  # intorno al freq converter error
}


# ─────────────────────────────────────────────────────────────────────────────
# LOAD un anno
# ─────────────────────────────────────────────────────────────────────────────

def load_year(year, turbine_id=TURBINE_ID):
    zpath = ZIPS[year]
    with zipfile.ZipFile(zpath) as z:
        fname = [f for f in z.namelist()
                 if f'Turbine_Data_Kelmarsh_{turbine_id}_' in f][0]
        with z.open(fname) as f:
            df = pd.read_csv(f, comment='#', header=None,
                             index_col=0, parse_dates=True)
    df.sort_index(inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE anno singolo
# ─────────────────────────────────────────────────────────────────────────────

def process_year(year):
    df = load_year(year)
    sensors_df, power_series, wind_series = extract_sensors(df)
    sensors_corr, dates = normalize_and_correct(sensors_df, power_series)
    t_dt, P, AR, noise_c, kap_c, comp_c = compute_P(sensors_corr, dates)
    return t_dt, P, AR, noise_c, kap_c, comp_c


# ─────────────────────────────────────────────────────────────────────────────
# ANALISI COMPLETA anno
# ─────────────────────────────────────────────────────────────────────────────

def analyze_year(year, t_dt, P, AR, thr=THR):
    faults   = T3_FAULTS.get(year, [])
    excludes = T3_EXCLUDE.get(year, [])

    # Periodi sani = tutto l'anno meno esclusioni e zone fault (±35gg)
    healthy_mask = np.ones(len(t_dt), dtype=bool)
    for ex_s, ex_e in excludes:
        healthy_mask &= ~((t_dt >= pd.Timestamp(ex_s)) &
                          (t_dt <= pd.Timestamp(ex_e)))
    for _, fdt, _, _ in faults:
        ft = pd.Timestamp(fdt)
        healthy_mask &= ~((t_dt >= ft - pd.Timedelta(days=40)) &
                          (t_dt <= ft + pd.Timedelta(days=10)))

    # Periodi sani contigui
    healthy_segs = []
    in_seg = False
    for i, m in enumerate(healthy_mask):
        if m and not in_seg:
            seg_start = t_dt[i]
            in_seg = True
        elif not m and in_seg:
            healthy_segs.append((seg_start, t_dt[i-1]))
            in_seg = False
    if in_seg:
        healthy_segs.append((seg_start, t_dt[-1]))

    # Conta FA in ogni segmento sano
    total_fa   = 0
    total_days = 0
    all_eps    = []
    for seg_s, seg_e in healthy_segs:
        n_fa, eps = count_alarms(t_dt, P, str(seg_s)[:19], str(seg_e)[:19], thr)
        days = (seg_e - seg_s).total_seconds() / 86400
        total_fa   += n_fa
        total_days += days
        all_eps.extend(eps)

    fa_per_month = total_fa / (total_days / 30.44) if total_days > 0 else 0.0

    # Advance warning per guasti
    detections = []
    for fname_f, fdt, dur_h, col in faults:
        from wind_turbine_precursor import measure_advance
        adv_P,  dt_P  = measure_advance(t_dt, P,  fdt, thr=thr, max_days=60)
        adv_AR, dt_AR = measure_advance(t_dt, AR, fdt, thr=thr, max_days=60)
        detections.append((fname_f, fdt, adv_P, adv_AR))

    return {
        'fa_total':     total_fa,
        'fa_per_month': fa_per_month,
        'healthy_days': total_days,
        'episodes':     all_eps,
        'detections':   detections,
        'healthy_segs': healthy_segs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else '.'

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║  SYMPHONON — Multi-anno T3 Kelmarsh 2016-2017-2018  ║")
    print(f"║  Soglia fissa: {THR}  (calibrata su 2016)              ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    yearly = {}
    for year in YEARS:
        print(f"── {year} ──")
        t_dt, P, AR, noise_c, kap_c, comp_c = process_year(year)
        res = analyze_year(year, t_dt, P, AR)
        yearly[year] = {
            't_dt': t_dt, 'P': P, 'AR': AR,
            'noise_c': noise_c, 'kap_c': kap_c, 'comp_c': comp_c,
            **res
        }

        # Stampa
        print(f"   Finestre: {len(t_dt)}  |  Giorni sani: {res['healthy_days']:.0f}  "
              f"({res['healthy_days']/30.44:.1f} mesi)")
        print(f"   FA totali: {res['fa_total']}  →  {res['fa_per_month']:.2f} FA/mese")

        if res['detections']:
            for fname_f, fdt, adv_P, adv_AR in res['detections']:
                s_P  = f"+{adv_P:.0f}gg"  if adv_P  else "—"
                s_AR = f"+{adv_AR:.0f}gg" if adv_AR else "—"
                print(f"   ✦ {fname_f}: P={s_P}  AR={s_AR}")
        else:
            print("   (nessun guasto meccanico — anno di validazione FA)")

        if res['episodes']:
            print(f"   Episodi FA: {len(res['episodes'])}")
            for ep_s, dur_h in res['episodes']:
                print(f"     {str(ep_s)[:16]}  durata={dur_h:.1f}h")
        print()

    # ── Riepilogo ────────────────────────────────────────────────────────────
    print("═"*65)
    print(f"  RIEPILOGO — Soglia P = {THR}")
    print("═"*65)
    print(f"  {'Anno':6s}  {'Tipo':14s}  {'FA/mese':9s}  {'FA totali':10s}  "
          f"{'Giorni sani':12s}  Detections")
    print("  " + "─"*61)

    total_fa_all   = 0
    total_days_all = 0
    for year in YEARS:
        r   = yearly[year]
        typ = "calibrazione" if year == 2016 else "validazione"
        det = ", ".join(
            f"{d[0]}={'+'+str(int(d[2]))+'gg' if d[2] else '—'}"
            for d in r['detections']
        ) or "—"
        print(f"  {year}   {typ:14s}  {r['fa_per_month']:7.2f}    "
              f"{r['fa_total']:4d}         {r['healthy_days']:6.0f} gg      {det}")
        if year != 2016:   # conta solo anni di validazione per FA totale
            total_fa_all   += r['fa_total']
            total_days_all += r['healthy_days']

    fa_overall = total_fa_all / (total_days_all / 30.44) if total_days_all else 0
    print("  " + "─"*61)
    print(f"  {'Validazione':21s}  {fa_overall:7.2f}    "
          f"{total_fa_all:4d}         {total_days_all:6.0f} gg  (2017+2018 combined)")
    print("═"*65)
    print()

    # ── PLOT timeline 3 anni ─────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=False)
    fig.patch.set_facecolor('#0a0a0a')

    colors_yr = {2016: '#edbd38', 2017: '#44aaff', 2018: '#88ff88'}

    for ax_idx, year in enumerate(YEARS):
        ax  = axes[ax_idx]
        r   = yearly[year]
        col = colors_yr[year]

        ax.set_facecolor('#111')
        ax.tick_params(colors='#aaa', labelsize=7)
        for sp in ax.spines.values(): sp.set_color('#333')

        # Segnale P
        ax.plot(r['t_dt'], r['P'], color=col, lw=1.1, alpha=0.9,
                label=f'P {year}')
        ax.plot(r['t_dt'], r['AR'], color='#aaaaff', lw=0.7,
                ls='--', alpha=0.4, label='AR')
        ax.axhline(THR, color=col, lw=0.8, ls=':', alpha=0.6,
                   label=f'Soglia {THR}')
        ax.set_ylim(-0.05, 1.15)
        ax.set_ylabel('P', color='#888', fontsize=8)

        # Ombreggia periodi sani
        for seg_s, seg_e in r['healthy_segs']:
            ax.axvspan(seg_s, seg_e, color='#225522', alpha=0.15)

        # Falsi allarmi
        for ep_s, dur_h in r['episodes']:
            ax.axvspan(ep_s,
                       ep_s + pd.Timedelta(hours=max(dur_h, 8)),
                       color='#ff8800', alpha=0.55, zorder=3, label='_nolegend_')
            ax.text(ep_s, 1.08, 'FA', color='#ff8800', fontsize=6,
                    ha='center', va='bottom')

        # Guasti reali
        for fname_f, fdt, adv_P, adv_AR in r['detections']:
            ft = pd.Timestamp(fdt)
            ax.axvline(ft, color='#ff3333', lw=1.8, ls='--', alpha=0.95)
            ax.fill_betweenx([0, 1],
                             ft - pd.Timedelta(days=35), ft,
                             color='#ff3333', alpha=0.08)
            ax.text(ft, 1.10, f'FAULT\n{fname_f[:12]}',
                    color='#ff3333', fontsize=6, ha='center', va='bottom')
            if adv_P:
                dt_alert = ft - pd.Timedelta(days=adv_P)
                ax.axvline(dt_alert, color=col, lw=1.0, ls=':', alpha=0.8)
                ax.text(dt_alert, 0.05, f'P alert\n−{adv_P:.0f}gg',
                        color=col, fontsize=6, ha='center', va='bottom',
                        bbox=dict(facecolor='#111', alpha=0.7, pad=2))

        # Formato asse X mensile
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

        fa_str = (f"{r['fa_per_month']:.2f} FA/mese ({r['fa_total']} in "
                  f"{r['healthy_days']:.0f}gg sani)")
        typ_str = "CALIBRAZIONE" if year == 2016 else "VALIDAZIONE"
        ax.set_title(f"{year} [{typ_str}]  —  {fa_str}",
                     color='#ddd', fontsize=9, pad=5)
        ax.legend(fontsize=7, labelcolor='#ccc', framealpha=0.15,
                  loc='upper right')

    plt.suptitle(f'T3 Kelmarsh 2016-2018 — Soglia P={THR} '
                 f'(verde=sano, arancio=FA, rosso=guasto reale)',
                 color='#ddd', fontsize=10, y=1.01)
    plt.tight_layout()
    out = f'{OUT_DIR}/wind_T3_multiyear.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
    plt.close()
    print(f"  Salvato: {out}")

    # ── PLOT barre riepilogo ─────────────────────────────────────────────────
    fig2, (ax_fa, ax_adv) = plt.subplots(1, 2, figsize=(11, 5))
    fig2.patch.set_facecolor('#0a0a0a')
    for ax in (ax_fa, ax_adv):
        ax.set_facecolor('#111')
        ax.tick_params(colors='#aaa', labelsize=9)
        for sp in ax.spines.values(): sp.set_color('#333')

    yrs  = [str(y) for y in YEARS]
    fa_r = [yearly[y]['fa_per_month'] for y in YEARS]
    cols = [colors_yr[y] for y in YEARS]
    types = ['calibrazione', 'validazione', 'validazione']

    bars = ax_fa.bar(yrs, fa_r, color=cols, alpha=0.8, width=0.5)
    ax_fa.axhline(1.0, color='#44cc88', lw=1.0, ls='--', alpha=0.8,
                  label='Soglia accettabilità (1/mese)')
    ax_fa.axhline(0.5, color='#88ff44', lw=0.8, ls=':', alpha=0.6,
                  label='Target ottimo (0.5/mese)')
    for bar, val, typ in zip(bars, fa_r, types):
        ax_fa.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.2f}\n({typ})', ha='center', va='bottom',
                   color='#ddd', fontsize=8)
    ax_fa.set_ylabel('False Alarms / mese', color='#888', fontsize=9)
    ax_fa.set_title('FA/mese per anno', color='#ddd', fontsize=10)
    ax_fa.legend(fontsize=8, labelcolor='#ccc', framealpha=0.2)
    ax_fa.set_ylim(0, max(fa_r) * 1.4 + 0.3)

    # Advance warning 2016
    adv_vals = []
    adv_labels = []
    for year in YEARS:
        for fname_f, fdt, adv_P, adv_AR in yearly[year]['detections']:
            adv_vals.append(adv_P or 0)
            adv_labels.append(f"{year}\n{fname_f[:15]}")

    if adv_vals:
        bar_adv = ax_adv.bar(range(len(adv_vals)), adv_vals,
                             color='#edbd38', alpha=0.85, width=0.5)
        ax_adv.axhline(30, color='#44cc88', lw=1.0, ls='--', alpha=0.8,
                       label='30gg (utilità minima)')
        ax_adv.axhline(14, color='#ff8844', lw=0.8, ls=':', alpha=0.6,
                       label='14gg (soglia critica)')
        for bar, val in zip(bar_adv, adv_vals):
            if val:
                ax_adv.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + 0.5,
                            f'+{val:.0f}gg', ha='center', va='bottom',
                            color='#edbd38', fontsize=9, fontweight='bold')
        ax_adv.set_xticks(range(len(adv_labels)))
        ax_adv.set_xticklabels(adv_labels, color='#aaa', fontsize=8)
        ax_adv.set_ylabel('Anticipo (giorni)', color='#888', fontsize=9)
        ax_adv.set_title('Advance warning guasti rilevati', color='#ddd', fontsize=10)
        ax_adv.legend(fontsize=8, labelcolor='#ccc', framealpha=0.2)
    else:
        ax_adv.text(0.5, 0.5, 'Nessun guasto\nrilevato in\n2017-2018',
                    transform=ax_adv.transAxes, ha='center', va='center',
                    color='#666', fontsize=11)
        ax_adv.set_title('Advance warning (nessun guasto 2017-2018)',
                         color='#ddd', fontsize=10)

    plt.suptitle(f'T3 Kelmarsh 2016-2018 — Sommario validazione (soglia={THR})',
                 color='#ddd', fontsize=10, y=1.01)
    plt.tight_layout()
    out2 = f'{OUT_DIR}/wind_T3_multiyear_summary.png'
    plt.savefig(out2, dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
    plt.close()
    print(f"  Salvato: {out2}")
    print("  Fatto.")
