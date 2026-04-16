"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  SYMPHONON — PRECURSOR SIGNAL su Dataset Bonn EEG                           ║
║                                                                              ║
║  Test falsificabile: il precursor_signal P distingue le classi EEG?         ║
║                                                                              ║
║  Dataset Bonn (Andrzejak et al. 2001):                                       ║
║    Z — sani, occhi aperti                                                    ║
║    O — sani, occhi chiusi                                                    ║
║    N — epilettici, inter-ictale, ippocampo opposto                           ║
║    F — epilettici, inter-ictale, zona epilettica                             ║
║    S — epilettici, ictale (crisi in corso)                                   ║
║                                                                              ║
║  Ipotesi: P(S) > P(F) > P(N) > P(O) ≈ P(Z)                                 ║
║                                                                              ║
║  Componenti precursor (mappatura EEG → Symphonon):                           ║
║    kap_norm    = rigidità oscillatoria (frequenza istantanea stabile)        ║
║    noise_norm  = sforzo verso sincronizzazione (ampiezza crescente)          ║
║    std_compress= compressione di fase (fasi convergenti = pre-ictale)        ║
║                                                                              ║
║  P = 0.45·noise + 0.30·kap + 0.25·compress  (pesi da Symphonon HTML)        ║
║                                                                              ║
║  Baseline EWS classici (confronto onesto):                                   ║
║    variance    = varianza del segnale nella finestra                         ║
║    autocorr    = autocorrelazione lag-1 (critical slowing down)              ║
║                                                                              ║
║  Download dataset:                                                           ║
║    https://www.ukbonn.de/epileptologie/arbeitsgruppen/                       ║
║    ag-lehnertz-neurophysik/downloads/                                        ║
║    Struttura attesa: data/bonn/{Z,O,N,F,S}/*.txt                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt
from scipy.stats import circvar, mannwhitneyu
from pathlib import Path
import sys, warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# COSTANTI
# ─────────────────────────────────────────────────────────────────────────────
CLASSES   = ['Z', 'O', 'N', 'F', 'S']
LABELS    = {
    'Z': 'Z\nsano\nocchi aperti',
    'O': 'O\nsano\nocchi chiusi',
    'N': 'N\ninter-ictale\nippoc. opposto',
    'F': 'F\ninter-ictale\nzona epil.',
    'S': 'S\nittale'
}
COLORS    = {'Z':'#2a9a4a','O':'#4a9a2a','N':'#9a8a2a','F':'#c47a18','S':'#cc2828'}
FS        = 173.61   # Hz — frequenza campionamento dataset Bonn
N_SAMPLES = 4096     # campioni per segmento (~23.6 secondi)
WIN_SEC   = 2.0      # finestra di analisi (secondi)
STEP_SEC  = 0.5      # passo scorrevole (secondi)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────

def load_class(data_dir, cls, max_files=100):
    """
    Carica tutti i segmenti .txt di una classe.
    Formato atteso: una colonna, 4096 righe per file.
    """
    cls_dir = Path(data_dir) / cls
    if not cls_dir.exists():
        return []
    files = sorted(list(cls_dir.glob('*.txt')) + list(cls_dir.glob('*.TXT')))[:max_files]
    segs = []
    for f in files:
        try:
            x = np.loadtxt(f, dtype=np.float32)
            if len(x) >= N_SAMPLES:
                segs.append(x[:N_SAMPLES])
        except Exception:
            pass
    return segs


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENTI PRECURSOR — singola finestra
# ─────────────────────────────────────────────────────────────────────────────

def _bandpass(x, lo=1.0, hi=40.0, fs=FS, order=4):
    """Filtro bandpass 1–40 Hz (banda EEG rilevante)."""
    nyq = fs / 2.0
    b, a = butter(order, [lo / nyq, hi / nyq], btype='band')
    return filtfilt(b, a, x)


def _components_window(seg):
    """
    Calcola le 3 componenti raw del precursor su un segmento (finestra).

    Returns: (kap, noise, compress, var_ews, ac_ews)
    """
    analytic  = hilbert(seg)
    phase     = np.angle(analytic)
    envelope  = np.abs(analytic)

    # ── KAP: rigidità oscillatoria ────────────────────────────────────
    # Frequenza istantanea = derivata della fase unwrapped
    # Alta rigidità = frequenza stabile (bassa varianza relativa)
    # Pre-ictale: oscillazioni diventano stereotipate → kap sale
    ifreq = np.diff(np.unwrap(phase)) * FS / (2 * np.pi)
    ifreq_pos = ifreq[ifreq > 0.5]  # solo freq plausibili
    if len(ifreq_pos) > 10:
        kap = 1.0 - np.std(ifreq_pos) / (np.mean(ifreq_pos) + 1e-8)
    else:
        kap = 0.0
    kap = float(np.clip(kap, 0.0, 1.0))

    # ── NOISE_NORM: sforzo verso sincronizzazione ─────────────────────
    # Ampiezza normalizzata rispetto alla baseline (1.0)
    # Durante la crisi l'ampiezza cresce molto sopra la baseline
    # → noise_norm sale quando il sistema "spinge" fuori dal regime normale
    env_mean = np.mean(envelope)
    env_std  = np.std(envelope)
    if env_mean > 1e-8:
        noise = float(np.clip(env_std / env_mean, 0.0, 3.0) / 3.0)
    else:
        noise = 0.0

    # ── STD_COMPRESS: compressione della distribuzione di fase ────────
    # Varianza circolare: 0 = fasi allineate, 1 = distribuzione uniforme
    # Pre-ictale: le fasi convergono → circvar scende → compress sale
    cv = float(circvar(phase))          # scipy: [0,1], 0=allineati
    compress = float(np.clip(1.0 - cv, 0.0, 1.0))

    # ── BASELINE EWS classici ─────────────────────────────────────────
    var_ews = float(np.var(seg))
    ac_ews  = float(np.corrcoef(seg[:-1], seg[1:])[0, 1]) if len(seg) > 1 else 0.0

    return kap, noise, compress, var_ews, ac_ews


# ─────────────────────────────────────────────────────────────────────────────
# PRECURSOR SU SEGMENTO COMPLETO (finestra scorrevole)
# ─────────────────────────────────────────────────────────────────────────────

def compute_precursor_segment(x, fs=FS, win_sec=WIN_SEC, step_sec=STEP_SEC):
    """
    Calcola P(t) su un segmento EEG con finestra scorrevole.

    Nota: le componenti vengono normalizzate LOCALMENTE al segmento
    per visualizzare le variazioni temporali. La normalizzazione
    GLOBALE (cross-classe) viene fatta in summarize_all().

    Returns:
        t     : array tempi (centro finestra, secondi)
        P     : precursor_signal P(t)
        raw   : dict con componenti raw (non normalizzate)
    """
    xf   = _bandpass(x, fs=fs)
    win  = int(win_sec * fs)
    step = int(step_sec * fs)
    N    = len(xf)

    t_arr, kap_arr, noise_arr, comp_arr = [], [], [], []
    var_arr, ac_arr = [], []

    for start in range(0, N - win, step):
        seg = xf[start:start + win]
        kap, noise, compress, var_ews, ac_ews = _components_window(seg)
        t_arr.append((start + win / 2) / fs)
        kap_arr.append(kap)
        noise_arr.append(noise)
        comp_arr.append(compress)
        var_arr.append(var_ews)
        ac_arr.append(ac_ews)

    t     = np.array(t_arr)
    kap   = np.array(kap_arr)
    noise = np.array(noise_arr)
    comp  = np.array(comp_arr)
    var_e = np.array(var_arr)
    ac_e  = np.array(ac_arr)

    # Normalizza localmente per il plot temporale
    def n01(a):
        mn, mx = a.min(), a.max()
        return (a - mn) / (mx - mn + 1e-12)

    P = 0.45 * n01(noise) + 0.30 * n01(kap) + 0.25 * n01(comp)

    return t, P, {'kap': kap, 'noise': noise, 'compress': comp,
                  'var': var_e, 'ac': ac_e}


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY CROSS-CLASSE (normalizzazione globale)
# ─────────────────────────────────────────────────────────────────────────────

def summarize_all(data_dir):
    """
    Calcola P_mean e baseline EWS per ogni classe con normalizzazione globale.

    Normalizzazione globale = le componenti vengono scalate su [0,1]
    usando min/max calcolati su TUTTI i segmenti di tutte le classi.
    Questo garantisce che il confronto tra classi sia onesto.
    """
    # Passo 1: raccolta raw
    raw_all = {cls: {'kap':[], 'noise':[], 'comp':[], 'var':[], 'ac':[]}
               for cls in CLASSES}
    loaded  = {}

    print("  Caricamento dati...")
    for cls in CLASSES:
        segs = load_class(data_dir, cls)
        if not segs:
            print(f"    {cls}: non trovato — skip")
            continue
        loaded[cls] = segs
        xf_list = [_bandpass(x) for x in segs]
        for xf in xf_list:
            win  = int(WIN_SEC * FS)
            step = int(STEP_SEC * FS)
            for start in range(0, len(xf) - win, step):
                seg = xf[start:start+win]
                kap, noise, comp, var_e, ac_e = _components_window(seg)
                raw_all[cls]['kap'].append(kap)
                raw_all[cls]['noise'].append(noise)
                raw_all[cls]['comp'].append(comp)
                raw_all[cls]['var'].append(var_e)
                raw_all[cls]['ac'].append(ac_e)
        print(f"    {cls}: {len(segs)} segmenti, "
              f"{len(raw_all[cls]['kap'])} finestre")

    if not loaded:
        return None, None

    # Passo 2: normalizzazione globale
    for feat in ['kap', 'noise', 'comp', 'var', 'ac']:
        all_vals = np.concatenate([np.array(raw_all[c][feat])
                                   for c in loaded])
        gmin, gmax = all_vals.min(), all_vals.max()
        for cls in loaded:
            a = np.array(raw_all[cls][feat])
            raw_all[cls][feat + '_n'] = (a - gmin) / (gmax - gmin + 1e-12)

    # Passo 3: P per finestra
    results = {}
    for cls in loaded:
        kn = raw_all[cls]['kap_n']
        nn = raw_all[cls]['noise_n']
        cn = raw_all[cls]['comp_n']
        vn = raw_all[cls]['var_n']
        an = raw_all[cls]['ac_n']

        P     = 0.45 * nn + 0.30 * kn + 0.25 * cn
        P_ews = 0.50 * vn + 0.50 * an  # baseline lineare

        results[cls] = {
            'P_mean':     float(np.mean(P)),
            'P_std':      float(np.std(P)),
            'P_max':      float(np.mean([np.max(P[i*20:(i+1)*20])
                          for i in range(len(P)//20) if i*20 < len(P)])),
            'EWS_mean':   float(np.mean(P_ews)),
            'EWS_std':    float(np.std(P_ews)),
            'P_all':      P,
            'EWS_all':    P_ews,
            'n_segs':     len(loaded[cls]),
        }

    return results, loaded


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICHE
# ─────────────────────────────────────────────────────────────────────────────

def print_stats(results):
    print()
    print("═" * 62)
    print("  RISULTATI — Precursor Signal vs EWS baseline")
    print("═" * 62)
    header = f"  {'Cls':4s}  {'P_mean':8s}  {'P_std':8s}  {'EWS_mean':10s}  {'n':5s}"
    print(header)
    print("  " + "─" * 58)
    for cls in CLASSES:
        if cls not in results:
            continue
        r = results[cls]
        print(f"  {cls:4s}  {r['P_mean']:8.4f}  {r['P_std']:8.4f}  "
              f"{r['EWS_mean']:10.4f}  {r['n_segs']:5d}")

    print()
    # Ranking
    ranked_P   = sorted([c for c in results], key=lambda c: results[c]['P_mean'], reverse=True)
    ranked_EWS = sorted([c for c in results], key=lambda c: results[c]['EWS_mean'], reverse=True)
    print(f"  Ranking P   : {' > '.join(ranked_P)}")
    print(f"  Ranking EWS : {' > '.join(ranked_EWS)}")
    print(f"  Atteso      : S > F > N > O > Z")

    # Test Mann-Whitney S vs Z
    if 'S' in results and 'Z' in results:
        stat, pval = mannwhitneyu(results['S']['P_all'],
                                  results['Z']['P_all'],
                                  alternative='greater')
        print()
        print(f"  Mann-Whitney P(S) > P(Z): U={stat:.0f}, p={pval:.4e}")
        if pval < 0.001:
            print("  → segnale statisticamente robusto (p<0.001)")
        elif pval < 0.05:
            print("  → segnale presente (p<0.05)")
        else:
            print("  → segnale debole o assente (p>0.05)")

    # Delta S-Z
    if 'S' in results and 'Z' in results:
        delta = results['S']['P_mean'] - results['Z']['P_mean']
        print(f"  Delta P(S-Z): {delta:+.4f}")
        verdict = ("FORTE" if abs(delta) > 0.15 else
                   "MODERATO" if abs(delta) > 0.05 else "DEBOLE")
        print(f"  Segnale: {verdict}")
    print("═" * 62)


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — Confronto classi (barre)
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(results, out_dir='.'):
    present = [c for c in CLASSES if c in results]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('#0a0a0a')

    titles = ['Symphonon Precursor Signal (P)', 'EWS Baseline (varianza + autocorr)']
    keys   = [('P_mean', 'P_std'), ('EWS_mean', 'EWS_std')]

    for ax, title, (mk, sk) in zip(axes, titles, keys):
        ax.set_facecolor('#111')
        ax.tick_params(colors='#aaa', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#333')

        means = [results[c][mk] for c in present]
        stds  = [results[c][sk] for c in present]
        cols  = [COLORS[c] for c in present]
        xlabs = [LABELS[c] for c in present]

        ax.bar(range(len(present)), means, yerr=stds,
               color=cols, alpha=0.85, width=0.6,
               error_kw={'color': '#666', 'capsize': 4, 'elinewidth': 1})
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels(xlabs, fontsize=7.5, color='#ccc')
        ax.set_ylabel('valore medio (norm. globale)', color='#aaa', fontsize=8)
        ax.set_title(title, color='#eee', fontsize=10, pad=10)
        ax.set_ylim(0, 1.0)
        ax.axhline(0.65, color='#cc5030', lw=0.9, ls='--',
                   alpha=0.6, label='soglia allerta')
        ax.legend(fontsize=7, labelcolor='#999', framealpha=0.2)

    plt.suptitle('Dataset Bonn EEG — Precursor Signal vs EWS baseline',
                 color='#ddd', fontsize=11, y=1.02)
    plt.tight_layout()
    out = Path(out_dir) / 'bonn_comparison.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
    plt.close()
    print(f"  Salvato: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — P(t) su singoli segmenti
# ─────────────────────────────────────────────────────────────────────────────

def plot_timeseries(loaded, n_each=3, out_dir='.'):
    present = [c for c in CLASSES if c in loaded]
    fig, axes = plt.subplots(len(present), 1, figsize=(14, 10), sharex=False)
    if len(present) == 1:
        axes = [axes]
    fig.patch.set_facecolor('#0a0a0a')

    for idx, cls in enumerate(present):
        ax = axes[idx]
        ax.set_facecolor('#0d0d0d')
        ax.tick_params(colors='#777', labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('#222')

        segs = loaded[cls][:n_each]
        for i, x in enumerate(segs):
            t, P, comp = compute_precursor_segment(x)
            offset = i * (t[-1] + 2.0)
            ax.plot(t + offset, P, color=COLORS[cls], alpha=0.85, lw=1.3)
            ax.fill_between(t + offset, P, alpha=0.12, color=COLORS[cls])

            # componenti sottili
            ax.plot(t + offset, 0.45 * comp['noise'] / (comp['noise'].max() + 1e-8),
                    color='#4488cc', alpha=0.3, lw=0.7, ls='--')
            ax.plot(t + offset, 0.30 * comp['kap'] / (comp['kap'].max() + 1e-8),
                    color='#44cc88', alpha=0.3, lw=0.7, ls='--')
            ax.plot(t + offset, 0.25 * comp['compress'] / (comp['compress'].max() + 1e-8),
                    color='#cc8844', alpha=0.3, lw=0.7, ls='--')

        ax.axhline(0.65, color='#cc4020', lw=0.7, ls='--', alpha=0.5)
        ax.set_ylim(-0.05, 1.1)
        ax.set_ylabel(cls, color=COLORS[cls], fontsize=11,
                      rotation=0, labelpad=20, fontweight='bold')
        if idx == len(present) - 1:
            ax.set_xlabel('tempo (s)', color='#888', fontsize=8)

    # legenda componenti
    axes[0].plot([], [], color='#4488cc', ls='--', alpha=0.6, lw=1, label='noise (ampiezza)')
    axes[0].plot([], [], color='#44cc88', ls='--', alpha=0.6, lw=1, label='kap (rigidità)')
    axes[0].plot([], [], color='#cc8844', ls='--', alpha=0.6, lw=1, label='compress (fase)')
    axes[0].plot([], [], color='white', lw=1.3, label='P (precursor)')
    axes[0].legend(fontsize=7, labelcolor='#aaa', framealpha=0.2,
                   loc='upper right')

    fig.suptitle('Symphonon P(t) — componenti per classe (3 segmenti per classe)',
                 color='#ddd', fontsize=10, y=1.01)
    plt.tight_layout()
    out = Path(out_dir) / 'bonn_timeseries.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
    plt.close()
    print(f"  Salvato: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 — Distribuzione P per classe (boxplot)
# ─────────────────────────────────────────────────────────────────────────────

def plot_distributions(results, out_dir='.'):
    present = [c for c in CLASSES if c in results]
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#111')
    ax.tick_params(colors='#aaa', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#333')

    data = [results[c]['P_all'] for c in present]
    cols = [COLORS[c] for c in present]

    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops={'color': 'white', 'lw': 2},
                    whiskerprops={'color': '#666'},
                    capprops={'color': '#666'},
                    flierprops={'marker': '.', 'alpha': 0.2, 'ms': 2})

    for patch, col in zip(bp['boxes'], cols):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)

    ax.set_xticks(range(1, len(present) + 1))
    ax.set_xticklabels([LABELS[c] for c in present], fontsize=8, color='#ccc')
    ax.set_ylabel('P (precursor signal)', color='#aaa', fontsize=9)
    ax.set_title('Distribuzione P per classe — normalizzazione globale',
                 color='#eee', fontsize=10, pad=10)
    ax.axhline(0.65, color='#cc5030', lw=0.9, ls='--',
               alpha=0.7, label='soglia allerta 0.65')
    ax.legend(fontsize=8, labelcolor='#999', framealpha=0.2)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    out = Path(out_dir) / 'bonn_distributions.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
    plt.close()
    print(f"  Salvato: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else 'data/bonn'
    OUT_DIR  = sys.argv[2] if len(sys.argv) > 2 else '.'

    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║  SYMPHONON — Precursor Signal su Dataset Bonn   ║")
    print("╚══════════════════════════════════════════════════╝")
    print()
    print(f"  Dataset : {DATA_DIR}")
    print(f"  Output  : {OUT_DIR}")
    print(f"  Finestra: {WIN_SEC}s  step: {STEP_SEC}s  FS: {FS}Hz")
    print()
    print("  Classi: Z O = sani | N F = inter-ictale | S = ictale")
    print("  Ipotesi: P(S) > P(F) > P(N) > P(O) ≈ P(Z)")
    print()

    results, loaded = summarize_all(DATA_DIR)

    if results is None:
        print()
        print("  Nessun dato trovato.")
        print()
        print("  Download dataset Bonn:")
        print("  https://www.ukbonn.de/epileptologie/arbeitsgruppen/")
        print("  ag-lehnertz-neurophysik/downloads/")
        print()
        print("  Struttura attesa:")
        print("  data/bonn/")
        print("    Z/  → Z001.txt ... Z100.txt")
        print("    O/  → O001.txt ... O100.txt")
        print("    N/  → N001.txt ... N100.txt")
        print("    F/  → F001.txt ... F100.txt")
        print("    S/  → S001.txt ... S100.txt")
        sys.exit(0)

    print_stats(results)

    print()
    print("── Plot ──")
    plot_comparison(results, OUT_DIR)
    plot_timeseries(loaded, n_each=3, out_dir=OUT_DIR)
    plot_distributions(results, OUT_DIR)

    print()
    print("  Fatto.")
