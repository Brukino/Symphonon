"""
PANDEMIC EARLY WARNING
Trajectory-Based Surveillance System — Hackathon Demo 2025

"Most systems wait until something is obvious.
 We detect when it becomes inevitable."
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
np.random.seed(42)
N            = 500
DEGRAD_START = 220
SPIKE_START  = 185
SPIKE_END    = 195

C = {
    "healthy"    : "#4fc3f7",
    "transient"  : "#ffd166",
    "degrad"     : "#ef476f",
    "bg"         : "#080e14",
    "panel"      : "#0e1520",
    "grid"       : "#1a2535",
    "text"       : "#d0d8e4",
    "subtext"    : "#6e7f94",
    "score"      : "#06d6a0",
    "classic"    : "#f07060",
    "warn_glow"  : "#ffdd57",
    "warn_label" : "#ffcc00",
    "arrow_degrad": "#ff6b8a",
    "arrow_noise" : "#4fc3f7",
}

# ─────────────────────────────────────────────────────────────
# 1. SYNTHETIC SIGNAL
# ─────────────────────────────────────────────────────────────
sig = np.zeros(N)
sig[:160]           = np.random.normal(0, 1.0, 160)
sig[160:DEGRAD_START] = np.random.normal(0, 1.0, 60)
sig[SPIKE_START:SPIKE_END] += (
    np.array([0.5, 1.5, 3.0, 4.2, 4.8, 4.2, 3.0, 1.5, 0.5, 0.1]) * 2.2
)
sig[DEGRAD_START:] = (
    np.linspace(0, 5.0, N - DEGRAD_START)
    + np.random.normal(0, 0.45, N - DEGRAD_START)
    + 0.3 * np.sin(np.linspace(0, 20, N - DEGRAD_START))
)
time = np.arange(N)

# ─────────────────────────────────────────────────────────────
# 2. CLASSIC DETECTOR — rolling z-score
# ─────────────────────────────────────────────────────────────
W_CLS = 30; THR_CLS = 3.5
classic_alerts = np.zeros(N)
for i in range(W_CLS, N):
    w   = sig[i - W_CLS : i]
    z   = (sig[i] - w.mean()) / (w.std() + 1e-9)
    if abs(z) > THR_CLS:
        classic_alerts[i] = 1

# ─────────────────────────────────────────────────────────────
# 3. CORE ENGINE — BLACK BOX
# Input: raw surveillance signal
# Output: latent trajectory (2D) + trajectory score [0,1]
# Internal mechanism not documented.
# ─────────────────────────────────────────────────────────────

_AF = 0.12   # internal constant
_AS = 0.04   # internal constant

def _latent(s):
    n = len(s)
    lx, ly = np.zeros(n), np.zeros(n)
    a, b = 0.0, 0.0
    for i in range(n):
        a = (1 - _AF) * a + _AF * s[i]
        b = (1 - _AS) * b + _AS * s[i]
        lx[i] = b
        ly[i] = (a - b) * 8.0
    return lx, ly

def _score(s):
    wl, wr = 20, 20
    n   = len(s)
    up  = np.zeros(n)
    for i in range(wl, n):
        up[i] = 1.0 if s[i] > s[i - wl] else 0.0
    sc = np.zeros(n)
    for i in range(wr, n):
        sc[i] = up[i - wr + 1 : i + 1].mean()
    return sc

latent_x, latent_y = _latent(sig)
raw_sc = _score(sig)

# smooth for display only
_vis = np.zeros(N); _v = 0.0
for i in range(N):
    _v = 0.75 * _v + 0.25 * raw_sc[i]; _vis[i] = _v

# alert trigger
_THR = 0.70; _PER = 5
system_alerts = np.zeros(N); _pc = 0
for i in range(N):
    _pc = _pc + 1 if raw_sc[i] > _THR else 0
    if _pc >= _PER: system_alerts[i] = 1

# ─────────────────────────────────────────────────────────────
# 4. DETECTION TIMES
# ─────────────────────────────────────────────────────────────
def _first(arr, t0, t1=None):
    t1 = t1 or len(arr)
    idx = np.where(arr[t0:t1] > 0)[0]
    return idx[0] + t0 if len(idx) else None

t_spike = _first(classic_alerts, SPIKE_START, SPIKE_END + 12)
t_cls   = _first(classic_alerts, DEGRAD_START)
t_trj   = _first(system_alerts,  DEGRAD_START)
advance = (t_cls - t_trj) if (t_cls and t_trj) else 0

phase_c = [
    C["healthy"] if i < 160 else
    C["transient"] if i < DEGRAD_START else
    C["degrad"]
    for i in range(N)
]

# ─────────────────────────────────────────────────────────────
# 5. FIGURE LAYOUT
# ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(19, 12.5), facecolor=C["bg"])

gs = gridspec.GridSpec(
    4, 2, figure=fig,
    left=0.055, right=0.975,
    top=0.895, bottom=0.06,
    hspace=0.55, wspace=0.34,
    height_ratios=[1.45, 1.15, 0.55, 0.22],
)

ax_sig  = fig.add_subplot(gs[0, :])
ax_lat  = fig.add_subplot(gs[1, 0])
ax_sc   = fig.add_subplot(gs[1, 1])
ax_cmp  = fig.add_subplot(gs[2, :])
ax_leg  = fig.add_subplot(gs[3, :])   # text comparison row

def _style(ax, hide_all=False):
    ax.set_facecolor(C["panel"])
    for sp in ax.spines.values():
        sp.set_edgecolor(C["grid"])
    if hide_all:
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor(C["bg"])
        for sp in ax.spines.values(): sp.set_visible(False)
    else:
        ax.tick_params(colors=C["subtext"], labelsize=8)
        ax.xaxis.label.set_color(C["subtext"])
        ax.yaxis.label.set_color(C["subtext"])
        ax.grid(color=C["grid"], lw=0.5, alpha=0.65)

for ax in [ax_sig, ax_lat, ax_sc, ax_cmp]: _style(ax)
_style(ax_leg, hide_all=True)

# ─────────────────────────────────────────────────────────────
# 6a. SIGNAL PANEL
# ─────────────────────────────────────────────────────────────
ax_sig.set_title("Surveillance Signal  (wastewater proxy)",
    color=C["text"], fontsize=11, fontweight="bold", pad=9, loc="left")

for i in range(1, N):
    ax_sig.plot(time[i-1:i+1], sig[i-1:i+1],
                color=phase_c[i], lw=1.25, alpha=0.92)

ax_sig.axvspan(0,           160,          alpha=0.05, color=C["healthy"],  zorder=0)
ax_sig.axvspan(160,         DEGRAD_START, alpha=0.07, color=C["transient"],zorder=0)
ax_sig.axvspan(DEGRAD_START, N,           alpha=0.05, color=C["degrad"],   zorder=0)

thr_abs = THR_CLS * sig[:160].std()
ax_sig.axhline( thr_abs, color=C["transient"], lw=0.9, ls="--", alpha=0.60,
                label=f"Classic threshold  (±{THR_CLS}σ)")
ax_sig.axhline(-thr_abs, color=C["transient"], lw=0.9, ls="--", alpha=0.60)

# Classic spike false alarm marker
if t_spike:
    ax_sig.annotate(
        "⚠  FALSE ALARM",
        xy=(t_spike, sig[t_spike]),
        xytext=(t_spike - 26, sig[t_spike] + 1.3),
        color=C["transient"], fontsize=8.5, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C["transient"], lw=1.1),
        zorder=10)

# Trajectory early-warning line — glowing
if t_trj:
    for lw, alpha in [(7, 0.10), (4, 0.20), (2, 0.60), (1.4, 1.0)]:
        ax_sig.axvline(t_trj, color=C["warn_glow"], lw=lw, alpha=alpha, zorder=8)
    ax_sig.text(
        t_trj + 3, sig.max() * 0.60,
        f"EARLY WARNING\nt = {t_trj}",
        color=C["warn_label"], fontsize=9, fontweight="bold", va="center",
        bbox=dict(boxstyle="round,pad=0.35", fc="#1a1400", ec=C["warn_label"],
                  alpha=0.88, lw=1.2),
        zorder=11)

# Classic late-alert line
if t_cls:
    ax_sig.axvline(t_cls, color=C["classic"], lw=1.4, alpha=0.80, ls="-",
                   label=f"Classic alert  t={t_cls}")
    ax_sig.text(
        t_cls + 3, sig.max() * 0.22,
        f"classic\nt = {t_cls}",
        color=C["classic"], fontsize=8, va="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="#1a0a08", ec=C["classic"],
                  alpha=0.80, lw=0.9),
        zorder=11)

# Phase labels
for x, lbl, col in [
    (80,  "HEALTHY",   C["healthy"]),
    (190, "TRANSIENT", C["transient"]),
    (362, "PROGRESSIVE\nDEGRADATION", C["degrad"]),
]:
    ax_sig.text(x, sig.max() * 0.875, lbl,
                color=col, fontsize=7.5, ha="center",
                fontweight="bold", alpha=0.82)

ax_sig.set_xlim(0, N)
ax_sig.set_ylabel("amplitude", fontsize=9)
ax_sig.legend(loc="upper left", fontsize=8, facecolor="#0e1520",
              edgecolor=C["grid"], labelcolor=C["text"], framealpha=0.92)

# ─────────────────────────────────────────────────────────────
# 6b. LATENT TRAJECTORY SPACE
# ─────────────────────────────────────────────────────────────
ax_lat.set_title("Latent Trajectory Space",
    color=C["text"], fontsize=11, fontweight="bold", pad=9, loc="left")

# Scatter by phase
ax_lat.scatter(latent_x[:160],             latent_y[:160],
    c=C["healthy"],  s=9,  alpha=0.40, zorder=2, label="Healthy")
ax_lat.scatter(latent_x[160:DEGRAD_START], latent_y[160:DEGRAD_START],
    c=C["transient"],s=11, alpha=0.60, zorder=3, label="Transient")
ax_lat.scatter(latent_x[DEGRAD_START:],    latent_y[DEGRAD_START:],
    c=C["degrad"],   s=11, alpha=0.65, zorder=3, label="Degradation")

# ── DEGRADATION ARROWS — large, visible, directional ──────
STRIDE_D = 10
for i in range(DEGRAD_START + STRIDE_D, N - STRIDE_D, STRIDE_D):
    dx = latent_x[i] - latent_x[i - STRIDE_D]
    dy = latent_y[i] - latent_y[i - STRIDE_D]
    mag = np.hypot(dx, dy)
    if mag < 0.001:
        continue
    ax_lat.annotate(
        "",
        xy     = (latent_x[i],           latent_y[i]),
        xytext = (latent_x[i - STRIDE_D], latent_y[i - STRIDE_D]),
        arrowprops=dict(
            arrowstyle   = "->",
            color        = C["arrow_degrad"],
            lw           = 1.8,
            alpha        = 0.80,
            mutation_scale = 14,
        ),
        zorder=6,
    )

# ── HEALTHY ARROWS — small, scattered, chaotic ────────────
STRIDE_H = 16
for i in range(30 + STRIDE_H, 155, STRIDE_H):
    dx = latent_x[i] - latent_x[i - STRIDE_H]
    dy = latent_y[i] - latent_y[i - STRIDE_H]
    mag = np.hypot(dx, dy)
    if mag < 0.001:
        continue
    ax_lat.annotate(
        "",
        xy     = (latent_x[i],           latent_y[i]),
        xytext = (latent_x[i - STRIDE_H], latent_y[i - STRIDE_H]),
        arrowprops=dict(
            arrowstyle   = "->",
            color        = C["arrow_noise"],
            lw           = 1.0,
            alpha        = 0.38,
            mutation_scale = 9,
        ),
        zorder=4,
    )

ax_lat.set_xlabel("latent_x", fontsize=9)
ax_lat.set_ylabel("latent_y", fontsize=9)

ax_lat.text(0.04, 0.10,
    "↻  chaotic\n    (noise)",
    transform=ax_lat.transAxes, color=C["healthy"],
    fontsize=8.5, alpha=0.90, va="bottom")

ax_lat.text(0.55, 0.83,
    "→  directed\n    (precursor)",
    transform=ax_lat.transAxes, color=C["arrow_degrad"],
    fontsize=9, fontweight="bold", alpha=0.97)

ax_lat.legend(
    loc="upper right", fontsize=7.5, facecolor="#0e1520",
    edgecolor=C["grid"], labelcolor=C["text"],
    markerscale=1.6, framealpha=0.92)

# ─────────────────────────────────────────────────────────────
# 6c. TRAJECTORY SCORE
# ─────────────────────────────────────────────────────────────
ax_sc.set_title("Trajectory Score  s(t)",
    color=C["text"], fontsize=11, fontweight="bold", pad=9, loc="left")

ax_sc.fill_between(time, _vis, alpha=0.18, color=C["score"])
ax_sc.plot(time, _vis, color=C["score"], lw=2.2)
ax_sc.axhline(_THR, color=C["warn_glow"], lw=1.0, ls="--", alpha=0.75,
              label="alert threshold")

ax_sc.axvspan(0,           160,          alpha=0.05, color=C["healthy"],  zorder=0)
ax_sc.axvspan(160,         DEGRAD_START, alpha=0.07, color=C["transient"],zorder=0)
ax_sc.axvspan(DEGRAD_START, N,           alpha=0.05, color=C["degrad"],   zorder=0)

# Early-warning glow on score panel
if t_trj:
    for lw, alpha in [(6, 0.10), (3, 0.22), (1.4, 0.90)]:
        ax_sc.axvline(t_trj, color=C["warn_glow"], lw=lw, alpha=alpha, zorder=8)

# Annotation: score stable during spike
if t_spike:
    spike_sc_val = _vis[t_spike]
    ax_sc.annotate(
        "stable during\ntransient spike",
        xy     = (t_spike, spike_sc_val),
        xytext = (t_spike - 62, 0.68),
        color  = C["subtext"], fontsize=7.5,
        arrowprops=dict(arrowstyle="->", color=C["subtext"], lw=0.8),
    )

ax_sc.set_xlim(0, N); ax_sc.set_ylim(-0.04, 1.06)
ax_sc.set_ylabel("trajectory score", fontsize=9)
ax_sc.legend(loc="upper left", fontsize=8, facecolor="#0e1520",
             edgecolor=C["grid"], labelcolor=C["text"], framealpha=0.92)

# ─────────────────────────────────────────────────────────────
# 6d. ALERT COMPARISON TIMELINE
# ─────────────────────────────────────────────────────────────
ax_cmp.set_title("Alert Timeline Comparison",
    color=C["text"], fontsize=10, fontweight="bold", pad=6, loc="left")

ax_cmp.axvspan(0,           160,          alpha=0.06, color=C["healthy"],  zorder=0)
ax_cmp.axvspan(160,         DEGRAD_START, alpha=0.08, color=C["transient"],zorder=0)
ax_cmp.axvspan(DEGRAD_START, N,           alpha=0.06, color=C["degrad"],   zorder=0)

for i in np.where(classic_alerts > 0)[0]:
    ax_cmp.axvline(i, ymin=0.55, ymax=0.94,
                   color=C["classic"], lw=0.85, alpha=0.80)
for i in np.where(system_alerts > 0)[0]:
    ax_cmp.axvline(i, ymin=0.06, ymax=0.45,
                   color=C["score"],   lw=1.0, alpha=0.92)

ax_cmp.set_xlim(0, N)
ax_cmp.set_yticks([0.25, 0.75])
ax_cmp.set_yticklabels(
    ["trajectory-based  (ours)", "classic  (amplitude)"],
    color=C["text"], fontsize=9)
ax_cmp.tick_params(axis="y", length=0)
ax_cmp.set_xlabel("Surveillance time  (hours)", fontsize=9)

if t_spike:
    ax_cmp.annotate(f"false alarm  t={t_spike}",
        xy=(t_spike, 0.75), xytext=(t_spike + 8, 0.75),
        color=C["transient"], fontsize=8,
        arrowprops=dict(arrowstyle="->", color=C["transient"], lw=0.9))

if t_trj:
    ax_cmp.annotate(f"⚡  EARLY WARNING  t={t_trj}",
        xy=(t_trj, 0.25), xytext=(t_trj + 14, 0.25),
        color=C["score"], fontsize=8.5, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C["score"], lw=1.0))

if t_cls:
    ax_cmp.annotate(f"late alert  t={t_cls}",
        xy=(t_cls, 0.75), xytext=(t_cls + 10, 0.75),
        color=C["classic"], fontsize=8,
        arrowprops=dict(arrowstyle="->", color=C["classic"], lw=0.9))

if t_trj and t_cls and advance > 0:
    ax_cmp.annotate("",
        xy=(t_cls, 0.50), xytext=(t_trj, 0.50),
        arrowprops=dict(arrowstyle="<->", color=C["warn_glow"],
                        lw=1.6, mutation_scale=12))
    ax_cmp.text(
        (t_trj + t_cls) / 2, 0.57,
        f"+{advance} hours earlier",
        ha="center", color=C["warn_glow"], fontsize=9, fontweight="bold")

# ─────────────────────────────────────────────────────────────
# 6e. BOTTOM TEXT COMPARISON BAR
# ─────────────────────────────────────────────────────────────
ax_leg.set_xlim(0, 1); ax_leg.set_ylim(0, 1)

# Divider
ax_leg.axhline(0.78, color=C["grid"], lw=0.8, alpha=0.6)

# Classic box
classic_txt = "Classic   →   reacts to peaks          (misses structured change)"
ax_leg.text(0.02, 0.35, classic_txt,
    color=C["classic"], fontsize=9.5, va="center", family="monospace",
    bbox=dict(boxstyle="round,pad=0.35", fc="#1a0a08",
              ec=C["classic"], alpha=0.75, lw=0.9))

# Ours box
ours_txt    = "Ours      →   detects consistent movement  (ignores isolated spikes)"
ax_leg.text(0.02, 0.35, " " * 200, alpha=0)   # spacer
ax_leg.text(0.51, 0.35, ours_txt,
    color=C["score"], fontsize=9.5, va="center", family="monospace",
    bbox=dict(boxstyle="round,pad=0.35", fc="#021a10",
              ec=C["score"], alpha=0.75, lw=0.9))

# ─────────────────────────────────────────────────────────────
# GLOBAL HEADER
# ─────────────────────────────────────────────────────────────
fig.text(0.50, 0.972,
    "PANDEMIC EARLY WARNING  ·  We detect direction, not spikes",
    ha="center", va="top",
    color=C["text"], fontsize=15, fontweight="bold",
    path_effects=[pe.withStroke(linewidth=4, foreground=C["bg"])])

fig.text(0.50, 0.950,
    "Most systems wait until something is obvious.  We detect when it becomes inevitable.",
    ha="center", va="top",
    color=C["subtext"], fontsize=10, style="italic")

# ── STATS BOX ──────────────────────────────────────────────
stats = (
    f"  Degradation onset : t = {DEGRAD_START}\n"
    f"  Classic false alarm: t = {t_spike}   [transient spike]\n"
    f"  Classic detection  : t = {t_cls}   [degradation]\n"
    f"  Trajectory warning : t = {t_trj}   [degradation]\n"
    f"  Advance            : +{advance} surveillance hours  "
)
fig.text(0.975, 0.065, stats,
    ha="right", va="bottom",
    color=C["subtext"], fontsize=7.8, family="monospace",
    bbox=dict(boxstyle="round,pad=0.5", fc=C["panel"],
              ec=C["grid"], alpha=0.92))

# ─────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────
for path in ["/mnt/user-data/outputs/pandemic_demo.png",
             "/home/claude/pandemic_demo.png"]:
    plt.savefig(path, dpi=165, facecolor=C["bg"], bbox_inches="tight")

print("✓  pandemic_demo.png saved")
print(f"\n   Degradation onset : t = {DEGRAD_START}")
print(f"   Classic false alarm: t = {t_spike}  [TRANSIENT SPIKE]")
print(f"   Classic detection  : t = {t_cls}  [DEGRADATION]")
print(f"   Trajectory warning : t = {t_trj}  [DEGRADATION]")
print(f"   ⚡ Advance          : +{advance} surveillance hours")
print(f"\n   Closing line:")
print(f"   'Most systems wait until something is obvious.")
print(f"    We detect when it becomes inevitable.'")
