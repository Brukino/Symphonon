"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  SYMPHONON — False Alarm Rate su Turbina 3 Kelmarsh 2016                   ║
║                                                                              ║
║  Guasto target: Oscillation encoder tower — 2016-03-01 (16h fermo)         ║
║  Anticipo misurato: ~31 giorni                                              ║
║                                                                              ║
║  Domanda: quante volte il segnale supera la soglia durante                  ║
║  operazione sana? Cioè: quanti falsi allarmi genererebbe?                  ║
║                                                                              ║
║  Metodo:                                                                     ║
║    - Periodo sano: aprile → ottobre (post-fault, pre-park-stop)             ║
║    - Definizione allarme: P > soglia per ≥2 finestre consecutive            ║
║    - Isteresi: allarme termina quando P < soglia − 0.08                     ║
║    - Varia soglia: 0.55 → 0.85 in step di 0.05                             ║
║    - Metriche: advance_warning(gg), FA/mese, precision, distanza da soglia  ║
║                                                                              ║
║  Confronto: Symphonon P vs AR (compress puro)                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import sys, warnings
warnings.filterwarnings('ignore')

# Importa da wind_turbine_precursor
sys.path.insert(0, '.')
from wind_turbine_precursor import (
    load_turbine, extract_sensors, normalize_and_correct, compute_P,
    measure_advance
)

FAULT_DT   = '2016-03-01 18:02'
# Periodo sano: aprile → fine ottobre (evita startup gen e park-stop nov)
HEALTHY_START = '2016-04-01'
HEALTHY_END   = '2016-10-30'

THRESHOLDS   = np.arange(0.55, 0.88, 0.05)
HYST         = 0.08    # isteresi: allarme off quando sig < thr - hyst
MIN_CONSEC   = 2       # finestre consecutive per confermare allarme
MAX_ADV_DAYS = 60      # finestra ricerca anticipo


# ─────────────────────────────────────────────────────────────────────────────
# CONTATORE ALLARMI con isteresi
# ─────────────────────────────────────────────────────────────────────────────

def count_alarms(t_dt, sig, start_dt, end_dt, thr, hyst=HYST, min_consec=MIN_CONSEC):
    """
    Conta episodi di allarme distinti nel periodo [start_dt, end_dt].

    Un episodio inizia quando sig >= thr per min_consec finestre consecutive.
    Termina quando sig < thr - hyst.

    Ritorna: (n_episodi, lista durate in ore, lista date di inizio)
    """
    mask = (t_dt >= pd.Timestamp(start_dt)) & (t_dt <= pd.Timestamp(end_dt))
    td   = t_dt[mask]
    sd   = sig[mask]

    n   = len(sd)
    in_alarm   = False
    consec     = 0
    episodes   = []
    ep_start   = None

    for i in range(n):
        v = sd[i]
        if not in_alarm:
            if v >= thr:
                consec += 1
                if consec >= min_consec:
                    in_alarm = True
                    ep_start = td[max(0, i - min_consec + 1)]
            else:
                consec = 0
        else:
            if v < thr - hyst:
                # allarme terminato
                ep_end = td[i]
                dur_h  = (ep_end - ep_start).total_seconds() / 3600
                episodes.append((ep_start, dur_h))
                in_alarm = False
                consec   = 0

    # Allarme ancora attivo alla fine
    if in_alarm and ep_start is not None:
        ep_end = td[-1]
        dur_h  = (ep_end - ep_start).total_seconds() / 3600
        episodes.append((ep_start, dur_h))

    return len(episodes), episodes


def months_in_period(start_dt, end_dt):
    s = pd.Timestamp(start_dt)
    e = pd.Timestamp(end_dt)
    return (e - s).days / 30.44


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else '.'

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║  FALSE ALARM RATE — Turbina 3 Kelmarsh 2016         ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    # Carica e calcola segnali
    df          = load_turbine(3)
    sensors_df, power_series, wind_series = extract_sensors(df)
    sensors_corr, dates = normalize_and_correct(sensors_df, power_series)
    t_dt, P, AR, noise_c, kap_c, comp_c = compute_P(sensors_corr, dates)

    n_months = months_in_period(HEALTHY_START, HEALTHY_END)
    print(f"  Periodo sano analizzato: {HEALTHY_START} → {HEALTHY_END}")
    print(f"  Durata: {n_months:.1f} mesi  ({int(n_months*30.44)} giorni)")
    print(f"  Finestre totali nel periodo sano: "
          f"{((t_dt >= pd.Timestamp(HEALTHY_START)) & (t_dt <= pd.Timestamp(HEALTHY_END))).sum()}")
    print()

    # ── Tabella soglie ────────────────────────────────────────────────────────
    print("═"*80)
    print(f"  {'Soglia':7s}  {'P adv(gg)':10s}  {'P FA/mese':10s}  "
          f"{'AR adv(gg)':10s}  {'AR FA/mese':10s}  Verdict")
    print("  " + "─"*76)

    results_P  = []
    results_AR = []

    for thr in THRESHOLDS:
        # Advance warning
        adv_P,  _ = measure_advance(t_dt, P,  FAULT_DT, thr=thr, max_days=MAX_ADV_DAYS)
        adv_AR, _ = measure_advance(t_dt, AR, FAULT_DT, thr=thr, max_days=MAX_ADV_DAYS)

        # False alarms nel periodo sano
        n_fa_P,  ep_P  = count_alarms(t_dt, P,  HEALTHY_START, HEALTHY_END, thr)
        n_fa_AR, ep_AR = count_alarms(t_dt, AR, HEALTHY_START, HEALTHY_END, thr)

        fa_rate_P  = n_fa_P  / n_months
        fa_rate_AR = n_fa_AR / n_months

        # Verdict
        if adv_P and n_fa_P <= 1:
            verd = "✓ P utile"
        elif adv_P and n_fa_P <= 3:
            verd = "~ P accettabile"
        elif adv_P:
            verd = "⚠ P troppi FA"
        else:
            verd = "✗ P nessun detect"

        s_P  = f"+{adv_P:.0f}gg"   if adv_P  else "—"
        s_AR = f"+{adv_AR:.0f}gg"  if adv_AR else "—"
        print(f"  {thr:.2f}     {s_P:10s}  {fa_rate_P:8.2f}/m   "
              f"{s_AR:10s}  {fa_rate_AR:8.2f}/m   {verd}")

        results_P.append({
            'thr': thr, 'adv': adv_P, 'fa': n_fa_P, 'fa_rate': fa_rate_P,
            'episodes': ep_P
        })
        results_AR.append({
            'thr': thr, 'adv': adv_AR, 'fa': n_fa_AR, 'fa_rate': fa_rate_AR,
            'episodes': ep_AR
        })

    print("═"*80)

    # ── Punto ottimale ───────────────────────────────────────────────────────
    print()
    print("  Punto ottimale (max anticipo con FA/mese ≤ 1.0):")
    for label, res in [("P", results_P), ("AR", results_AR)]:
        candidates = [(r['thr'], r['adv'], r['fa_rate'])
                      for r in res if r['adv'] and r['fa_rate'] <= 1.0]
        if candidates:
            best = max(candidates, key=lambda x: x[1])
            print(f"    {label}: soglia={best[0]:.2f}  anticipo=+{best[1]:.0f}gg  "
                  f"FA/mese={best[2]:.2f}")
        else:
            print(f"    {label}: nessun punto con FA/mese ≤ 1.0")

    # ── Dettaglio episodi al punto ottimale ─────────────────────────────────
    print()
    best_thr_P = next((r for r in results_P if r['adv'] and r['fa_rate'] <= 1.0), None)
    if best_thr_P is None:
        best_thr_P = min(results_P, key=lambda r: r['fa_rate'])

    print(f"  Episodi P a soglia={best_thr_P['thr']:.2f} nel periodo sano:")
    if best_thr_P['episodes']:
        for ep_start, dur_h in best_thr_P['episodes']:
            print(f"    {str(ep_start)[:16]}  durata={dur_h:.1f}h")
    else:
        print("    nessuno")

    # ── Plot ─────────────────────────────────────────────────────────────────
    thrs      = [r['thr']     for r in results_P]
    adv_P_arr = [r['adv'] or 0 for r in results_P]
    fa_P_arr  = [r['fa_rate'] for r in results_P]
    adv_A_arr = [r['adv'] or 0 for r in results_AR]
    fa_A_arr  = [r['fa_rate'] for r in results_AR]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#0a0a0a')
    for ax in axes:
        ax.set_facecolor('#111')
        ax.tick_params(colors='#aaa', labelsize=8)
        for sp in ax.spines.values(): sp.set_color('#333')

    # 1. Advance warning vs soglia
    axes[0].plot(thrs, adv_P_arr, 'o-', color='#edbd38', lw=1.5, label='P')
    axes[0].plot(thrs, adv_A_arr, 's--', color='#aaaaff', lw=1.2, label='AR')
    axes[0].axhline(30, color='#44cc88', lw=0.8, ls=':', alpha=0.7,
                    label='30gg (soglia utilità)')
    axes[0].set_xlabel('Soglia', color='#888'); axes[0].set_ylabel('Anticipo (giorni)', color='#888')
    axes[0].set_title('Advance Warning vs soglia', color='#ddd', fontsize=9)
    axes[0].legend(fontsize=8, labelcolor='#ccc', framealpha=0.2)

    # 2. FA/mese vs soglia
    axes[1].plot(thrs, fa_P_arr, 'o-', color='#edbd38', lw=1.5, label='P')
    axes[1].plot(thrs, fa_A_arr, 's--', color='#aaaaff', lw=1.2, label='AR')
    axes[1].axhline(1.0, color='#44cc88', lw=0.8, ls=':', alpha=0.7,
                    label='1 FA/mese (accettabile)')
    axes[1].set_xlabel('Soglia', color='#888'); axes[1].set_ylabel('FA/mese', color='#888')
    axes[1].set_title('False Alarms/mese vs soglia', color='#ddd', fontsize=9)
    axes[1].legend(fontsize=8, labelcolor='#ccc', framealpha=0.2)

    # 3. Tradeoff: anticipo vs FA/mese (curva ROC-like)
    axes[2].plot(fa_P_arr, adv_P_arr, 'o-', color='#edbd38', lw=1.5, label='P')
    axes[2].plot(fa_A_arr, adv_A_arr, 's--', color='#aaaaff', lw=1.2, label='AR')
    for r in results_P:
        if r['adv']:
            axes[2].annotate(f"{r['thr']:.2f}",
                             (r['fa_rate'], r['adv']),
                             textcoords='offset points', xytext=(4,3),
                             fontsize=6.5, color='#edbd38')
    axes[2].axhline(30, color='#44cc88', lw=0.8, ls=':', alpha=0.7)
    axes[2].axvline(1.0, color='#ff6644', lw=0.8, ls=':', alpha=0.7)
    axes[2].set_xlabel('FA/mese', color='#888')
    axes[2].set_ylabel('Anticipo (giorni)', color='#888')
    axes[2].set_title('Tradeoff: anticipo ↔ falsi allarmi\n(↑ sinistra = meglio)',
                      color='#ddd', fontsize=9)
    axes[2].legend(fontsize=8, labelcolor='#ccc', framealpha=0.2)

    # ── Plot timeline con allarmi ─────────────────────────────────────────
    # Usa la soglia ottimale P
    opt_thr = best_thr_P['thr']
    fig2, (ax_sig, ax_fa) = plt.subplots(2, 1, figsize=(15, 7),
                                          gridspec_kw={'height_ratios': [2, 1]})
    fig2.patch.set_facecolor('#0a0a0a')
    for ax in (ax_sig, ax_fa):
        ax.set_facecolor('#111')
        ax.tick_params(colors='#aaa', labelsize=7)
        for sp in ax.spines.values(): sp.set_color('#333')

    ax_sig.plot(t_dt, P,  color='#edbd38', lw=1.2, label='Symphonon P')
    ax_sig.plot(t_dt, AR, color='#aaaaff', lw=0.8, ls='--', alpha=0.7, label='AR')
    ax_sig.axhline(opt_thr, color='#edbd38', lw=0.7, ls=':', alpha=0.6,
                   label=f'Soglia ottimale={opt_thr:.2f}')

    # Zona sana
    ax_sig.axvspan(pd.Timestamp(HEALTHY_START), pd.Timestamp(HEALTHY_END),
                   color='#44aa44', alpha=0.06, label='Periodo sano analizzato')
    # Zona pre-fault (30 gg)
    ft = pd.Timestamp(FAULT_DT)
    ax_sig.axvspan(ft - pd.Timedelta(days=35), ft, color='#ff4444', alpha=0.10,
                   label='Pre-fault (35gg)')
    ax_sig.axvline(ft, color='#ff4444', lw=1.5, ls='--')
    ax_sig.text(ft, 1.05, 'FAULT', color='#ff4444', fontsize=7, ha='center')

    # Falsi allarmi nel periodo sano
    for ep_start, dur_h in best_thr_P['episodes']:
        ax_sig.axvspan(ep_start,
                       ep_start + pd.Timedelta(hours=max(dur_h, 4)),
                       color='#ff8800', alpha=0.35, zorder=3)

    ax_sig.set_ylim(-0.05, 1.15)
    ax_sig.set_title(f'T3 Kelmarsh 2016 — P signal, soglia={opt_thr:.2f}  '
                     f'(arancio=falsi allarmi, rosso=pre-fault)',
                     color='#ddd', fontsize=9)
    ax_sig.legend(fontsize=7, labelcolor='#ccc', framealpha=0.2, loc='upper right')
    ax_sig.set_ylabel('P / AR', color='#888', fontsize=8)

    # Componenti
    ax_fa.plot(t_dt, noise_c, color='#ff8844', lw=0.8, alpha=0.8, label='noise')
    ax_fa.plot(t_dt, kap_c,   color='#88ff44', lw=0.8, alpha=0.8, label='kap')
    ax_fa.plot(t_dt, comp_c,  color='#44aaff', lw=0.8, alpha=0.8, label='compress')
    ax_fa.axvline(ft, color='#ff4444', lw=1.0, ls='--', alpha=0.7)
    ax_fa.set_ylim(-0.05, 1.10)
    ax_fa.set_ylabel('Componenti', color='#888', fontsize=7)
    ax_fa.legend(fontsize=7, labelcolor='#ccc', framealpha=0.2)

    plt.tight_layout()
    out2 = f'{OUT_DIR}/wind_T3_false_alarm_timeline.png'
    plt.savefig(out2, dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
    plt.close()

    plt.figure(fig.number)
    plt.suptitle('T3 Kelmarsh — Tradeoff anticipo vs falsi allarmi',
                 color='#ddd', fontsize=10, y=1.01)
    plt.tight_layout()
    out1 = f'{OUT_DIR}/wind_T3_false_alarm_tradeoff.png'
    plt.savefig(out1, dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
    plt.close()

    print(f"\n  Salvato: {out1}")
    print(f"  Salvato: {out2}")
    print("  Fatto.")
