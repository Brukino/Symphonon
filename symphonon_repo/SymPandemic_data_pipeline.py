# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════╗
║         SYMPHONON PANDEMIC EWS  —  Early Warning System             ║
║         Multi-Signal Surveillance & Trajectory Detection             ║
╠══════════════════════════════════════════════════════════════════════╣
║  Version  : 1.0.0-alpha                                              ║
║  Tag      : symphonon-pandemic-ews-v1.0.0-alpha                      ║
║  Track    : AI x Biosecurity Hackathon 2026 — Pandemic Early Warning ║
╠══════════════════════════════════════════════════════════════════════╣
║  Architecture                                                        ║
║  ─────────────────────────────────────────────────────────────────  ║
║  Signal ingestion  : NWSS wastewater (national + spatial)            ║
║                    + OWID cases / deaths / excess mortality          ║
║  Quality layer     : per-signal coverage fraction + gap flags        ║
║  Normalisation     : rolling 104-week p95 (strictly causal)          ║
║  Engine            : per-signal trajectory scoring (direction,       ║
║                      not magnitude) → quality-weighted fusion        ║
║  Dual alert system :                                                 ║
║    [1] Fusion channel   — magnitude trajectory, 4 signals            ║
║    [2] Spatial channel  — geographic spread, node-count gated        ║
║  Validation        : 5 COVID waves, FPR on inter-wave periods        ║
╠══════════════════════════════════════════════════════════════════════╣
║  Results                                                             ║
║    See run output — numbers depend on data version and bug fixes.    ║
║    Do not use hardcoded values from this header.                     ║
╠══════════════════════════════════════════════════════════════════════╣
║  Data sources                                                        ║
║    NWSS : https://data.cdc.gov/api/views/2ew6-ywp6/rows.csv         ║
║    OWID : https://covid.ourworldindata.org/data/owid-covid-data.csv  ║
╠══════════════════════════════════════════════════════════════════════╣
║  Usage                                                               ║
║    python SymPandemic_data_pipeline.py            # real data        ║
║    python SymPandemic_data_pipeline.py --offline  # synthetic mode   ║
║    python SymPandemic_data_pipeline.py --inspect  # inspect CSVs     ║
║    python SymPandemic_data_pipeline.py --refresh  # force re-download║
╚══════════════════════════════════════════════════════════════════════╝
"""

import argparse
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
warnings.filterwarnings("ignore")

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
NWSS_URL  = "https://data.cdc.gov/api/views/2ew6-ywp6/rows.csv?accessType=DOWNLOAD"
# Primary OWID URL (new catalog, may change):
OWID_URL  = "https://catalog.ourworldindata.org/garden/covid/latest/compact/compact.csv"
# Stable fallback URL (keep in sync):
OWID_URL_FALLBACK = "https://covid.ourworldindata.org/data/owid-covid-data.csv"

NWSS_CACHE = "nwss_raw.csv"
OWID_CACHE = "owid_raw.csv"

C = {
    "bg"      : "#080e14",
    "panel"   : "#0e1520",
    "grid"    : "#1a2535",
    "text"    : "#d0d8e4",
    "subtext" : "#6e7f94",
    "ww"      : "#4fc3f7",   # wastewater
    "cases"   : "#ffd166",   # confirmed cases
    "excess"  : "#ef476f",   # excess mortality
    "score"   : "#06d6a0",   # trajectory score
    "warn"    : "#ffdd57",
}

# Spatial spread accent colour — used in both COVID and flu plots
ACCENT = "#00e5ff"

# -------------------------------------------------------------
# 1. DATA ACQUISITION
# -------------------------------------------------------------

def download(url, cache_path, label, refresh=False):
    """Download with cache. Returns path to local file."""
    import os, requests
    if os.path.exists(cache_path) and not refresh:
        print(f"  [{label}] using cache: {cache_path}")
        return cache_path
    if refresh and os.path.exists(cache_path):
        os.remove(cache_path)
        print(f"  [{label}] cache cleared, re-downloading...")
    print(f"  [{label}] downloading from {url[:60]}...")
    r = requests.get(url, timeout=180, stream=True)
    r.raise_for_status()
    with open(cache_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    size = os.path.getsize(cache_path) / 1e6
    print(f"  [{label}] saved {cache_path}  ({size:.1f} MB)")
    return cache_path


def inspect_csvs(nwss_path, owid_path):
    """Print column names, dtypes and 3 sample rows for both CSVs."""
    for label, path in [("NWSS", nwss_path), ("OWID", owid_path)]:
        print(f"\n" + "-"*60)
        print(f"  {label}  ->  {path}")
        df = pd.read_csv(path, nrows=5, low_memory=False)
        print(f"  columns ({len(df.columns)}):")
        for c in df.columns:
            print(f"    {c:<45} {str(df[c].dtype):<10}  sample: {df[c].iloc[0]}")
    print(f"\n" + "-"*60)


def load_nwss(path):
    """
    Load NWSS wastewater metric data preserving geographic granularity.

    Returns a dict with:
      "national"  : weekly national median (as before)
      "by_state"  : weekly median per state (wide: columns = state names)
      "spatial"   : weekly spatial dispersion score — fraction of states
                    trending upward, independent of magnitude.
                    This is the key early-warning signal: a pandemic spreads
                    geographically before it shows up in national aggregates.
    """
    df = pd.read_csv(path, low_memory=False)
    print(f"  [NWSS] raw shape: {df.shape}")

    # Date column
    date_col = (
        next((c for c in df.columns if c.lower() == "date_end"),  None)
        or next((c for c in df.columns if c.lower() == "date_start"), None)
        or next((c for c in df.columns if "date" in c.lower()),   None)
    )
    if date_col is None:
        raise ValueError("No date column found in NWSS data")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.rename(columns={date_col: "date"})
    df["week"] = df["date"].dt.to_period("W").dt.start_time

    # Geographic column
    geo_col = next((c for c in df.columns
                    if c.lower() in ("wwtp_jurisdiction", "state",
                                     "reporting_jurisdiction")), None)
    if geo_col is None:
        print("  [NWSS] WARNING: no geo column found, falling back to national only")

    # Value column
    pct_col  = next((c for c in df.columns if "percentile"    in c.lower()), None)
    ptc_col  = next((c for c in df.columns if "ptc_15d"       in c.lower()), None)
    conc_col = next((c for c in df.columns if "concentration" in c.lower()), None)
    val_col  = pct_col or ptc_col or conc_col
    if val_col is None:
        raise ValueError(f"No usable value column. Available: {list(df.columns)}")

    print(f"  [NWSS] using value={val_col}  geo={geo_col}")
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df[val_col] = df[val_col].replace(-99, np.nan)

    col_max = df[val_col].quantile(0.99)
    if col_max > 1.5:
        print(f"  [NWSS] rescaling {val_col} from 0-{col_max:.0f} to 0-1")
        df[val_col] = df[val_col] / 100.0

    # ── National weekly median ─────────────────────────────────
    national = (
        df.groupby("week")[val_col]
        .median()
        .reset_index()
        .rename(columns={val_col: "ww_signal"})
        .sort_values("week")
    )
    print(f"  [NWSS] national: {len(national)} weeks  "
          f"({national['week'].min().date()} -> {national['week'].max().date()})")

    if geo_col is None:
        return {"national": national, "by_state": None, "spatial": None}

    # ── State-level weekly median ──────────────────────────────
    # Exclude territories and non-state entries for cleaner signal
    exclude = {"VI", "GU", "PR", "MP", "AS", "USA", "US"}
    df_states = df[~df[geo_col].isin(exclude)].copy()

    by_state = (
        df_states.groupby(["week", geo_col])[val_col]
        .median()
        .unstack(geo_col)
        .sort_index()
    )
    n_states = by_state.shape[1]
    print(f"  [NWSS] states: {n_states} jurisdictions  "
          f"{list(by_state.columns[:6])}...")

    # ── Spatial dispersion score ───────────────────────────────
    # At each week, for each state: 1 if current value > value 4 weeks ago.
    # Spatial score = fraction of states trending up.
    # Leading interpretation: geographic spread precedes national magnitude.
    LAG = 4
    trending_up = (by_state > by_state.shift(LAG)).astype(float)
    # Require at least 10 states reporting to compute a meaningful fraction
    reporting   = by_state.notna().sum(axis=1)
    spatial     = trending_up.sum(axis=1) / reporting.clip(lower=1)
    spatial     = spatial.where(reporting >= 10, np.nan)

    spatial_df = (
        pd.DataFrame({
            "ww_spatial"         : spatial,
            "n_states_reporting" : reporting,
        })
        .reset_index()
        .rename(columns={"index": "week"})
    )
    # fix column name if index wasn't named "index"
    if "week" not in spatial_df.columns:
        spatial_df = spatial_df.rename(columns={spatial_df.columns[0]: "week"})

    print(f"  [NWSS] spatial score: {spatial_df['ww_spatial'].notna().sum()} "
          f"valid weeks  (requires >=10 states reporting)")

    return {"national": national, "by_state": by_state, "spatial": spatial_df}


def load_owid(path):
    """
    Load OWID compact COVID dataset, filter USA.
    Extract: new_cases_smoothed_per_million, new_deaths_smoothed_per_million,
             excess_mortality (% above baseline).
    Resample to weekly.
    """
    df = pd.read_csv(path, low_memory=False)
    print(f"  [OWID] raw shape: {df.shape}")

    # Detect country/location column
    loc_col = next((c for c in df.columns if c.lower() in ("location", "country", "entity")), None)
    date_col = next((c for c in df.columns if c.lower() == "date"), None)
    if loc_col is None or date_col is None:
        raise ValueError(f"Cannot identify columns. Available: {list(df.columns[:20])}")

    df = df[df[loc_col] == "United States"].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).rename(columns={date_col: "date"})
    df = df.sort_values("date")

    cols_want = {
        "new_cases_smoothed_per_million"  : "cases_pm",
        "new_deaths_smoothed_per_million" : "deaths_pm",
        "excess_mortality"                : "excess_mort",
    }
    for src, dst in cols_want.items():
        if src in df.columns:
            df[dst] = pd.to_numeric(df[src], errors="coerce")
        else:
            print(f"  [OWID] column not found: {src}")
            df[dst] = np.nan

    df["week"] = df["date"].dt.to_period("W").dt.start_time

    weekly = (
        df.groupby("week")[list(cols_want.values())]
        .mean()
        .reset_index()
        .sort_values("week")
    )
    print(f"  [OWID] USA weekly series: {len(weekly)} weeks  "
          f"({weekly['week'].min().date()} → {weekly['week'].max().date()})")
    return weekly



# Gaps longer than this trigger a warning and suppress the signal locally
GAP_WARN_WEEKS = 4
# Rolling window for quality score (how far back we look to assess coverage)
QUALITY_WINDOW  = 12


def align(nwss_data, owid):
    """
    Merge NWSS + OWID on week, adding the spatial dispersion score
    as an independent signal alongside the national wastewater median.

    For each signal, before interpolating:
      - record which weeks have real observations  -> <signal>_present
      - compute rolling coverage fraction          -> <signal>_quality  [0,1]
      - flag runs of consecutive gaps > threshold  -> <signal>_gap
    """
    nwss     = nwss_data["national"]
    spatial  = nwss_data["spatial"]

    # Merge national + spatial + OWID
    merged = pd.merge(nwss, owid, on="week", how="outer")
    if spatial is not None:
        merged = pd.merge(merged, spatial, on="week", how="left")
    merged = merged.sort_values("week").set_index("week")

    signals = ["ww_signal", "ww_spatial", "cases_pm", "deaths_pm", "excess_mort"]

    # n_states_reporting is kept separate — it's a count, not a signal,
    # and must NOT be interpolated (missing = truly no states reporting).
    if "n_states_reporting" in merged.columns:
        merged["n_states_reporting"] = merged["n_states_reporting"].fillna(0).astype(int)
    else:
        merged["n_states_reporting"] = 0

    print(f"\n  [ALIGN] date range: "
          f"{merged.index.min().date()} → {merged.index.max().date()}  "
          f"({len(merged)} weeks)")

    for s in signals:
        if s not in merged.columns:
            merged[s] = np.nan

        raw = merged[s].copy()

        # ── 1. presence mask (before any fill) ───────────────────
        present = raw.notna().astype(float)
        merged[f"{s}_present"] = present

        # ── 2. rolling quality score ──────────────────────────────
        quality = present.rolling(QUALITY_WINDOW, min_periods=1).mean()
        merged[f"{s}_quality"] = quality

        # ── 3. gap flag: consecutive NaN runs > threshold ─────────
        gap_flag = pd.Series(False, index=raw.index)
        in_gap, gap_len = False, 0
        for idx, val in raw.items():
            if pd.isna(val):
                gap_len += 1
                if gap_len >= GAP_WARN_WEEKS:
                    in_gap = True
            else:
                in_gap, gap_len = False, 0
            gap_flag[idx] = in_gap
        merged[f"{s}_gap"] = gap_flag.astype(bool)

        # ── 4. report ─────────────────────────────────────────────
        n_total   = len(raw)
        n_real    = int(present.sum())
        n_missing = n_total - n_real
        pct       = 100 * n_real / n_total
        mean_q    = quality.mean()
        gap_weeks = int(gap_flag.sum())
        status    = "OK" if pct >= 80 else ("WARN" if pct >= 50 else "POOR")
        print(f"  [ALIGN] {s:<18} {n_real:>4}/{n_total} real weeks "
              f"({pct:5.1f}%)  mean_quality={mean_q:.2f}  "
              f"gap_weeks={gap_weeks}  [{status}]")

        # ── 5. interpolate then fill edges ────────────────────────
        merged[s] = raw.interpolate("linear").ffill().bfill()

    # Clip negative excess mortality (P-score artefact below baseline)
    if "excess_mort" in merged.columns:
        merged["excess_mort"] = merged["excess_mort"].clip(lower=0)

    # Normalise each signal to [0,1] via rolling 104-week trailing p95.
    # Strictly causal (no center=True).
    # ww_spatial is excluded: it is already in [0,1] by construction
    # (fraction of states trending up). Applying p95 normalisation would
    # distort its absolute meaning — 70% states trending up becoming 1.0
    # is not the same as it being a "maximum" event.
    NORM_WINDOW = 104
    signals_to_normalise = [s for s in signals if s != "ww_spatial"]
    for s in signals_to_normalise:
        col = merged[s]
        p95 = col.rolling(NORM_WINDOW, min_periods=8).quantile(0.95)
        p95 = p95.ffill().bfill()
        p95 = p95.replace(0, np.nan).ffill().bfill()
        merged[f"{s}_norm"] = (col / p95).clip(0, 1)

    # ww_spatial: copy directly, already in [0,1]
    if "ww_spatial" in merged.columns:
        merged["ww_spatial_norm"] = merged["ww_spatial"].clip(0, 1)

    return merged


# -------------------------------------------------------------
# 2. CORE ENGINE
# Each signal has its own temporal profile.
# Scoring happens per-signal first; fusion operates on scores.
# -------------------------------------------------------------

_AF = 0.12
_AS = 0.04

# Per-signal configuration.
# wl  : look-back window for direction test (weeks)
# wr  : consistency window — fraction of weeks trending up (weeks)
# smooth_alpha : EMA smoothing. Lower = more reactive, higher = more stable.
# weight : fusion weight (renormalised at runtime over available signals)
#
# Rationale:
#   wastewater   → short windows, low smoothing  → reacts fast, accepts noise
#   cases        → medium windows                → mid-lag, moderate smoothing
#   deaths       → longer windows, high smoothing → lagging, needs stability
#   excess_mort  → longest windows               → slowest, most confirmation-like
# Signals that enter the magnitude fusion.
# ww_spatial is intentionally excluded — it measures geographic spread,
# not magnitude, and has a separate alert channel (see evaluate()).
SIGNAL_CONFIG = {
    "ww_signal_norm"   : dict(wl=4,  wr=4,  smooth_alpha=0.45, weight=0.45),
    "cases_pm_norm"    : dict(wl=6,  wr=6,  smooth_alpha=0.55, weight=0.30),
    "deaths_pm_norm"   : dict(wl=10, wr=8,  smooth_alpha=0.70, weight=0.15),
    "excess_mort_norm" : dict(wl=12, wr=10, smooth_alpha=0.75, weight=0.10),
}

# Spatial spread detector — independent channel.
# Shorter windows: geographic diffusion is faster than magnitude build-up.
# Higher quality threshold: sparse state coverage produces unstable fractions.
SPATIAL_CONFIG = dict(wl=3, wr=3, smooth_alpha=0.38)
SPATIAL_MIN_QUALITY = 0.65   # stricter than ALERT_MIN_QUALITY


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


def _score(s, wl=8, wr=8):
    """
    Trajectory score for a single normalised signal.
    At each step: 1 if s[t] > s[t-wl], else 0.
    Score = fraction of 1s in the trailing wr-week window.
    Returns array in [0, 1].

    Burn-in: first (wl + wr) weeks have score = 0 — insufficient history.
    Callers should be aware that the system cannot alert during this period.
    For default COVID params (wl=4, wr=4): burn-in = 8 weeks.
    For flu params (wl=3, wr=3): burn-in = 6 weeks.
    """
    n = len(s)
    up = np.array([1.0 if i >= wl and s[i] > s[i - wl] else 0.0
                   for i in range(n)])
    sc = np.zeros(n)
    for i in range(wr, n):
        sc[i] = up[i - wr + 1: i + 1].mean()
    return sc


def _smooth(sc, alpha=0.6):
    """Exponential moving average. alpha controls memory (higher = more lag)."""
    vis = np.zeros(len(sc))
    v = 0.0
    for i, x in enumerate(sc):
        v = alpha * v + (1 - alpha) * x
        vis[i] = v
    return vis


# CSD window: how many weeks of history to use for AR1 and variance estimation.
# Shorter = more reactive, noisier. Longer = more stable, slower to respond.
# 52 weeks (1 year) gives enough pre-transition history for robust estimation.
CSD_WIN = 52


def _csd_score(s, win=None):
    """
    Critical Slowing Down score for a single signal.

    Before a critical transition (bifurcation), a dynamical system exhibits:
      1. Rising variance — larger fluctuations as the restoring force weakens
      2. Rising lag-1 autocorrelation (AR1) — slower recovery from perturbations

    Both are computed on the DETRENDED signal (remove rolling mean first) to
    separate genuine CSD from upward trend. Detrending is causal (rolling mean
    uses only past data, no centering).

    Output: score in [0,1], combining normalised AR1 and normalised variance.
    Burn-in: first `win` weeks are 0 — insufficient history.

    Known limitations:
      - Requires approximately stationary system between transitions.
        Strong external forcing (seasonal flu, endemic COVID) can produce
        spurious CSD signals unrelated to bifurcation.
      - Sensitive to the detrending window choice. Too short = residuals
        still contain trend; too long = slow to adapt.
      - Most theoretically valid for novel pandemic emergence (no prior baseline).
        Less valid for seasonal pathogens with strong periodic forcing.

    Returns array in [0, 1].
    """
    if win is None:
        win = CSD_WIN

    n   = len(s)
    ar1 = np.zeros(n)
    var = np.zeros(n)

    for i in range(win, n):
        window = s[i - win: i]

        # ── 1. Causal detrending: subtract rolling mean ───────────
        trend    = np.mean(window)
        residual = window - trend

        # ── 2. AR1: lag-1 autocorrelation on residuals ───────────
        # Pearson correlation between residual[t] and residual[t-1]
        if residual.std() > 1e-9:
            r = residual[:-1]
            r1 = residual[1:]
            # Only compute if variance is non-trivial
            corr = np.corrcoef(r, r1)[0, 1]
            ar1[i] = max(0.0, corr)   # clip negative — we care about slowing, not speeding
        else:
            ar1[i] = 0.0

        # ── 3. Variance of detrended residuals ────────────────────
        var[i] = residual.var()

    # ── Normalise both to [0,1] via rolling max (causal) ─────────
    # Use expanding max — each week's value is relative to its own history.
    # This is causal and avoids anchoring to the global max (unknown in real-time).
    def _rolling_max_norm(x):
        out = np.zeros(n)
        running_max = 0.0
        for i in range(n):
            if x[i] > running_max:
                running_max = x[i]
            out[i] = x[i] / running_max if running_max > 1e-12 else 0.0
        return out

    ar1_norm = _rolling_max_norm(ar1)
    var_norm  = _rolling_max_norm(var)

    # ── Combine: equal weight, smooth lightly ─────────────────────
    combined = 0.5 * ar1_norm + 0.5 * var_norm
    score    = _smooth(combined, alpha=0.40)

    return np.clip(score, 0, 1)


def compute_trajectory(merged):
    """
    Two independent scoring pipelines:

    1. FUSION — magnitude signals (ww_signal, cases, deaths, excess_mort).
       Scores fused with quality-adjusted weights → fusion_score.

    2. SPATIAL — geographic spread signal (ww_spatial).
       Scored independently → spatial_score.
       Not mixed into fusion_score. Evaluated as a separate alert channel.
    """
    individual_scores = {}

    # ── Pipeline 1: magnitude fusion ──────────────────────────
    for col, cfg in SIGNAL_CONFIG.items():
        if col not in merged.columns:
            print(f"  [ENGINE] signal not found, skipping: {col}")
            continue
        s  = merged[col].values
        sc = _score(s, wl=cfg["wl"], wr=cfg["wr"])
        sc = _smooth(sc, alpha=cfg["smooth_alpha"])
        individual_scores[col] = sc
        merged[f"score_{col}"] = sc
        print(f"  [ENGINE] scored {col:<25}  wl={cfg['wl']} wr={cfg['wr']}  "
              f"α={cfg['smooth_alpha']}  w={cfg['weight']}")

    fusion    = np.zeros(len(merged))
    total_eff = np.zeros(len(merged))
    for col, sc in individual_scores.items():
        base_w = SIGNAL_CONFIG[col]["weight"]
        q_col  = col.replace("_norm", "_quality")
        # Use per-week quality (not global mean) so sparse early periods
        # are automatically down-weighted at the weeks they are sparse,
        # not across the entire time series.
        if q_col in merged.columns:
            q_vec = merged[q_col].values
        else:
            q_vec = np.ones(len(merged))
        eff_w      = base_w * q_vec          # shape: (n,)
        fusion    += eff_w * sc
        total_eff += eff_w
        mean_q = float(q_vec.mean())
        print(f"  [ENGINE] {col:<25}  base_w={base_w:.2f}  "
              f"mean_quality={mean_q:.2f}  (applied per-week)")

    # Avoid division by zero in weeks where all signals have zero quality
    safe_total = np.where(total_eff > 0, total_eff, 1.0)
    fusion /= safe_total
    merged["fusion_score"] = fusion
    print(f"  [ENGINE] fusion over {len(individual_scores)} signals  "
          f"(per-week quality weighting)")

    # ── Pipeline 2: spatial spread (independent) ──────────────
    if "ww_spatial_norm" in merged.columns:
        sp = merged["ww_spatial_norm"].values
        sc = _score(sp, wl=SPATIAL_CONFIG["wl"], wr=SPATIAL_CONFIG["wr"])
        sc = _smooth(sc, alpha=SPATIAL_CONFIG["smooth_alpha"])
        merged["spatial_score"] = sc
        print(f"  [ENGINE] spatial_score  wl={SPATIAL_CONFIG['wl']} "
              f"wr={SPATIAL_CONFIG['wr']}  α={SPATIAL_CONFIG['smooth_alpha']}  "
              f"[independent channel]")
    else:
        merged["spatial_score"] = np.nan
        print(f"  [ENGINE] spatial_score  not available")

    # ── Pipeline 3: Critical Slowing Down (independent) ───────
    if "ww_signal_norm" in merged.columns:
        csd = _csd_score(merged["ww_signal_norm"].values)
        merged["csd_score"] = csd
        print(f"  [ENGINE] csd_score  AR1+variance on ww_signal  "
              f"[independent channel, win={CSD_WIN}w]")
    else:
        merged["csd_score"] = np.nan
        print(f"  [ENGINE] csd_score  not available (no ww_signal_norm)")

    return merged, individual_scores


# -------------------------------------------------------------
# 3. VALIDATION FRAMEWORK
# -------------------------------------------------------------

# Known COVID-19 wave onsets in the USA (approximate week dates)
KNOWN_WAVES = {
    "Alpha"   : "2021-03-01",
    "Delta"   : "2021-07-05",
    "Omicron" : "2021-12-06",
    "BA.5"    : "2022-06-27",
    "XBB"     : "2022-12-05",
}

# Weeks after onset to consider "in-wave" (alerts here are true positives)
WAVE_ACTIVE_WEEKS = 12

# Adaptive threshold parameters
# Alert fires when fusion_score > baseline + ADAPT_K * local_std
# baseline = rolling mean over ADAPT_WIN weeks
# This replaces the fixed 0.65 threshold
ADAPT_WIN = 26   # 6-month baseline window
ADAPT_K   = 2.0  # increased from 1.5 — more conservative, fewer false positives
ADAPT_MIN_THR = 0.30  # floor: never alert below this absolute level

# Minimum data quality required to fire an alert.
# If the leading signal (wastewater) has rolling quality below this,
# the alert is suppressed regardless of the fusion score.
# Prevents alerting on interpolated/filled gaps masquerading as signal.
ALERT_MIN_QUALITY = 0.50


def _adaptive_threshold(scores, win=ADAPT_WIN, k=ADAPT_K, min_thr=ADAPT_MIN_THR):
    """
    Time-varying alert threshold.
    thr[t] = max(min_thr, mean(scores[t-win:t]) + k * std(scores[t-win:t]))

    Burn-in: first `win` weeks use min_thr (fixed floor). During burn-in
    the comparison with the classic detector is not valid — exclude from FPR.

    Known trade-off: the threshold is computed from the same signal being
    thresholded. During a genuine wave, as the score rises the threshold also
    rises — reducing sensitivity at the peak. This is intentional (avoids
    continuous alerts through an entire wave) but means the detector is most
    sensitive at the ONSET, not the peak, which is correct for early warning.
    """
    n   = len(scores)
    thr = np.full(n, min_thr)
    for i in range(win, n):
        w      = scores[i - win: i]
        local  = w.mean() + k * w.std()
        thr[i] = max(min_thr, local)
    return thr


def evaluate(merged, persist=3):
    """
    Evaluate trajectory detector vs classic z-score detector.

    Uses adaptive threshold instead of fixed value.
    Also measures false positive rate during inter-wave periods.

    Returns:
      results      : DataFrame with per-wave detection summary
      traj_alert   : binary alert array (trajectory)
      classic_alert: binary alert array (classic z-score)
      thr_arr      : adaptive threshold array
      fp_report    : dict with false positive stats
    """
    scores = merged["fusion_score"].values
    dates  = merged.index
    n      = len(scores)

    # ── Quality gate for fusion alert (wastewater magnitude) ──
    ww_q      = merged.get("ww_signal_quality",
                           pd.Series(1.0, index=merged.index)).values
    ww_quality = ww_q  # fusion gate: only ww_signal quality

    # ── Adaptive threshold + fusion trajectory alert ───────────
    thr_arr    = _adaptive_threshold(scores)
    pc         = 0
    traj_alert = np.zeros(n)
    for i in range(n):
        quality_ok = ww_quality[i] >= ALERT_MIN_QUALITY
        above_thr  = scores[i] > thr_arr[i]
        pc = pc + 1 if (above_thr and quality_ok) else 0
        if pc >= persist:
            traj_alert[i] = 1

    # ── Spatial spread alert (independent channel) ────────────
    # Gate: fires only when enough states are actively reporting THIS week.
    # This is a network-density check at observation time, not a historical
    # coverage fraction — the signal is valid when the network is dense now.
    SPATIAL_MIN_NODES = 20   # minimum states reporting to trust spatial score

    sp_scores = merged["spatial_score"].values if "spatial_score" in merged.columns \
                else np.zeros(n)
    n_nodes   = merged["n_states_reporting"].values if "n_states_reporting" in merged.columns \
                else np.zeros(n)
    sp_thr    = _adaptive_threshold(sp_scores, k=ADAPT_K)
    pc_sp     = 0
    spatial_alert = np.zeros(n)
    for i in range(n):
        nodes_ok     = n_nodes[i] >= SPATIAL_MIN_NODES
        sp_above_thr = sp_scores[i] > sp_thr[i]
        pc_sp = pc_sp + 1 if (sp_above_thr and nodes_ok) else 0
        if pc_sp >= persist:
            spatial_alert[i] = 1

    # ── CSD alert — independent channel ──────────────────────────
    # CSD fires when AR1+variance score is anomalously high.
    # Uses a stricter k (k=1.5) because CSD is inherently noisier than
    # trajectory scoring, and we want to detect genuine slowing-down,
    # not baseline fluctuations.
    # Requires ww_signal quality >= ALERT_MIN_QUALITY to be meaningful
    # (CSD on interpolated gaps is noise by definition).
    csd_scores = merged["csd_score"].values if "csd_score" in merged.columns \
                 else np.zeros(n)
    csd_thr = _adaptive_threshold(csd_scores, k=1.5)
    pc_csd  = 0
    csd_alert = np.zeros(n)
    for i in range(n):
        csd_ok = csd_scores[i] > csd_thr[i] and ww_quality[i] >= ALERT_MIN_QUALITY
        pc_csd = pc_csd + 1 if csd_ok else 0
        if pc_csd >= persist:
            csd_alert[i] = 1
    cas = merged["cases_pm_norm"].values
    classic_alert = np.zeros(n)
    W = 8
    for i in range(W, n):
        w = cas[i - W: i]
        z = (cas[i] - w.mean()) / (w.std() + 1e-9)
        if z > 2.5:
            classic_alert[i] = 1

    # ── Per-wave detection timing ──────────────────────────────
    # Search 8 weeks before onset to reveal alerts already firing early,
    # rather than masking them by starting the window exactly at onset.
    SEARCH_BEFORE = 8
    results = []
    for name, onset_str in KNOWN_WAVES.items():
        onset = pd.Timestamp(onset_str)
        if onset < dates.min() or onset > dates.max():
            continue
        idx_onset = dates.searchsorted(onset, side="left")
        idx0 = max(0, idx_onset - SEARCH_BEFORE)
        idx1 = min(idx_onset + WAVE_ACTIVE_WEEKS, n)

        traj_hits = np.where(traj_alert[idx0:idx1])[0]
        sp_hits   = np.where(spatial_alert[idx0:idx1])[0]
        csd_hits  = np.where(csd_alert[idx0:idx1])[0]
        cls_hits  = np.where(classic_alert[idx0:idx1])[0]

        t_traj    = dates[idx0 + traj_hits[0]] if len(traj_hits) else None
        t_spatial = dates[idx0 + sp_hits[0]]   if len(sp_hits)   else None
        t_csd     = dates[idx0 + csd_hits[0]]  if len(csd_hits)  else None
        t_cls     = dates[idx0 + cls_hits[0]]  if len(cls_hits)  else None

        adv_traj = int((t_cls - t_traj).days / 7)    if (t_traj and t_cls)    else None
        adv_sp   = int((t_cls - t_spatial).days / 7) if (t_spatial and t_cls) else None
        adv_csd  = int((t_cls - t_csd).days / 7)     if (t_csd and t_cls)     else None

        results.append({
            "wave"        : name,
            "onset"       : onset.date(),
            "traj"        : t_traj.date()    if t_traj    else "—",
            "spatial"     : t_spatial.date() if t_spatial else "—",
            "csd"         : t_csd.date()     if t_csd     else "—",
            "classic"     : t_cls.date()     if t_cls     else "—",
            "adv_traj"    : adv_traj if adv_traj is not None else "—",
            "adv_spatial" : adv_sp   if adv_sp   is not None else "—",
            "adv_csd"     : adv_csd  if adv_csd  is not None else "—",
        })

    # ── False positive rates ───────────────────────────────────
    # Exclude burn-in period (first ADAPT_WIN weeks) from FPR calculation:
    # during burn-in the adaptive threshold is fixed at ADAPT_MIN_THR,
    # which is not comparable to the classic detector's calibration.
    burn_in = np.zeros(n, dtype=bool)
    burn_in[:ADAPT_WIN] = True

    # in_wave covers both pre-onset search window and active wave period
    in_wave    = np.zeros(n, dtype=bool)
    for onset_str in KNOWN_WAVES.values():
        onset = pd.Timestamp(onset_str)
        if onset < dates.min() or onset > dates.max():
            continue
        idx_onset = dates.searchsorted(onset, side="left")
        idx0 = max(0, idx_onset - SEARCH_BEFORE)
        idx1 = min(idx_onset + WAVE_ACTIVE_WEEKS, n)
        in_wave[idx0:idx1] = True

    inter_wave = ~in_wave & ~burn_in

    fp_report = {
        "inter_wave_weeks" : int(inter_wave.sum()),
        "fp_traj"          : int(traj_alert[inter_wave].sum()),
        "fp_spatial"       : int(spatial_alert[inter_wave].sum()),
        "fp_csd"           : int(csd_alert[inter_wave].sum()),
        "fp_classic"       : int(classic_alert[inter_wave].sum()),
        "fpr_traj"         : round(traj_alert[inter_wave].mean(), 3),
        "fpr_spatial"      : round(spatial_alert[inter_wave].mean(), 3),
        "fpr_csd"          : round(csd_alert[inter_wave].mean(), 3),
        "fpr_classic"      : round(classic_alert[inter_wave].mean(), 3),
    }

    return (pd.DataFrame(results), traj_alert, spatial_alert, csd_alert,
            classic_alert, thr_arr, sp_thr, csd_thr, fp_report)


# -------------------------------------------------------------
# 4. SYNTHETIC FALLBACK (offline / dev mode)
# -------------------------------------------------------------

def make_synthetic():
    """Generate realistic multi-signal synthetic data for 3 years weekly."""
    print("  [SYNTHETIC] generating 3-year weekly multi-signal data...")
    np.random.seed(42)
    N    = 156  # ~3 years weekly
    weeks = pd.date_range("2021-01-04", periods=N, freq="W")

    def _wave(t, center, width, height):
        return height * np.exp(-0.5 * ((t - center) / width) ** 2)

    t = np.arange(N)

    # Wastewater: leading by ~2 weeks relative to cases
    ww = (np.random.normal(0, 0.08, N)
          + _wave(t,  10, 4, 0.9)   # Alpha
          + _wave(t,  28, 5, 1.0)   # Delta
          + _wave(t,  50, 6, 1.5)   # Omicron
          + _wave(t,  78, 4, 0.7)   # BA.5
          + _wave(t, 100, 5, 0.8))  # XBB

    # Cases: peaks ~2w after wastewater
    cases = (np.random.normal(0, 0.06, N)
             + _wave(t,  12, 4, 0.85)
             + _wave(t,  30, 5, 0.95)
             + _wave(t,  52, 6, 1.40)
             + _wave(t,  80, 4, 0.65)
             + _wave(t, 102, 5, 0.75))

    # Deaths: peaks ~4w after wastewater
    deaths = (np.random.normal(0, 0.04, N)
              + _wave(t,  14, 5, 0.70)
              + _wave(t,  32, 5, 0.80)
              + _wave(t,  54, 7, 1.20)
              + _wave(t,  82, 4, 0.55)
              + _wave(t, 104, 5, 0.60))

    # Spatial score: fraction of states trending up — slightly leads national
    spatial = (np.random.normal(0, 0.06, N)
               + _wave(t,   9, 4, 0.85)
               + _wave(t,  27, 5, 0.90)
               + _wave(t,  49, 6, 1.30)
               + _wave(t,  77, 4, 0.65)
               + _wave(t,  99, 5, 0.75))

    # Excess mortality: smoothest, most lagging
    excess = (np.random.normal(0, 0.03, N)
              + _wave(t,  15, 5, 0.60)
              + _wave(t,  33, 5, 0.70)
              + _wave(t,  55, 7, 1.10)
              + _wave(t,  83, 4, 0.50))

    def _norm(x):
        """
        Synthetic normalisation: clip to [0,inf] and divide by max.
        NOTE: real pipeline uses rolling 104-week p95 — results are NOT
        directly comparable. Synthetic mode is for smoke-testing only.
        """
        x = np.clip(x, 0, None)
        mx = x.max() or 1
        return x / mx

    df = pd.DataFrame({
        "ww_signal_norm"       : _norm(ww),
        "ww_spatial_norm"      : _norm(spatial),
        "cases_pm_norm"        : _norm(cases),
        "deaths_pm_norm"       : _norm(deaths),
        "excess_mort_norm"     : _norm(excess),
        "ww_signal"            : _norm(ww),
        "ww_spatial"           : _norm(spatial),
        "cases_pm"             : _norm(cases),
        "deaths_pm"            : _norm(deaths),
        "excess_mort"          : _norm(excess),
        # Quality = 1.0 for all synthetic signals (no gaps by construction)
        "ww_signal_quality"    : 1.0,
        "ww_spatial_quality"   : 1.0,
        "cases_pm_quality"     : 1.0,
        "deaths_pm_quality"    : 1.0,
        "excess_mort_quality"  : 1.0,
        # Synthetic: assume all 51 states reporting throughout
        "n_states_reporting"   : 51,
    }, index=weeks)
    df.index.name = "week"
    # Compute CSD score on synthetic ww_signal (will be mostly 0 — no real
    # pre-transition slowing in synthetic Gaussian waves, which is correct)
    df["csd_score"] = _csd_score(df["ww_signal_norm"].values)
    return df


# -------------------------------------------------------------
# 5. VISUALISATION
# -------------------------------------------------------------

def plot(merged, results, traj_alert, spatial_alert, csd_alert,
         classic_alert, thr_arr, fp_report, mode="real"):
    idx = merged.index
    if hasattr(idx, "to_timestamp"):
        dates = idx.to_timestamp()
    else:
        dates = pd.DatetimeIndex(idx)

    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    # ── Layout: 7 rows × 2 cols ────────────────────────────────
    # Row 0      : multi-signal surveillance     (full width)
    # Rows 1-3   : 5 latent trajectory panels   (2×2 + 1 paired with fusion)
    #              [ww_national | ww_spatial   ]
    #              [cases       | deaths        ]
    #              [excess_mort | fusion score  ]  ← fusion beside last latent
    # Row 4      : alert timeline                (full width)
    # Row 5      : results table + FPR           (full width)
    fig = plt.figure(figsize=(20, 26), facecolor=C["bg"])
    gs  = gridspec.GridSpec(6, 2, figure=fig,
                            left=0.06, right=0.97,
                            top=0.94, bottom=0.03,
                            hspace=0.50, wspace=0.28,
                            height_ratios=[1.2, 0.80, 0.80, 0.80, 0.72, 0.40])

    ax_sig   = fig.add_subplot(gs[0, :])
    ax_lat   = {
        "ww_signal_norm"   : fig.add_subplot(gs[1, 0]),
        "ww_spatial_norm"  : fig.add_subplot(gs[1, 1]),
        "cases_pm_norm"    : fig.add_subplot(gs[2, 0]),
        "deaths_pm_norm"   : fig.add_subplot(gs[2, 1]),
        "excess_mort_norm" : fig.add_subplot(gs[3, 0]),
    }
    ax_sc    = fig.add_subplot(gs[3, 1])   # fusion score beside excess_mort
    ax_cmp   = fig.add_subplot(gs[4, :])
    ax_tbl   = fig.add_subplot(gs[5, :])

    def _style(ax):
        ax.set_facecolor(C["panel"])
        for sp in ax.spines.values(): sp.set_edgecolor(C["grid"])
        ax.tick_params(colors=C["subtext"], labelsize=8)
        ax.grid(color=C["grid"], lw=0.5, alpha=0.6)
        ax.xaxis.label.set_color(C["subtext"])
        ax.yaxis.label.set_color(C["subtext"])

    for ax in list(ax_lat.values()) + [ax_sig, ax_sc, ax_cmp]:
        _style(ax)

    # ── 5a. Multi-signal panel ─────────────────────────────────
    ax_sig.set_title("Multi-Signal Surveillance  (normalised)",
                     color=C["text"], fontsize=11, fontweight="bold", pad=8, loc="left")

    sig_cfg = [
        ("ww_signal_norm",    C["ww"],     "wastewater national (NWSS)",  2.2),
        ("ww_spatial_norm",   "#00e5ff",   "wastewater spatial spread",   1.6),
        ("cases_pm_norm",     C["cases"],  "cases/million (OWID)",        1.4),
        ("deaths_pm_norm",    C["excess"], "deaths/million (OWID)",       1.0),
        ("excess_mort_norm",  "#c77dff",   "excess mortality (OWID)",     0.9),
    ]
    for col, color, label, lw in sig_cfg:
        if col in merged.columns:
            ax_sig.plot(dates, merged[col].values,
                        color=color, lw=lw, alpha=0.85, label=label)

    for name, onset_str in KNOWN_WAVES.items():
        onset = pd.Timestamp(onset_str)
        if merged.index.min() <= onset <= merged.index.max():
            ax_sig.axvline(onset, color=C["warn"], lw=0.8, ls=":", alpha=0.6)
            ax_sig.text(onset, 0.98, name, color=C["warn"],
                        fontsize=7, ha="center", va="top",
                        transform=ax_sig.get_xaxis_transform())

    ax_sig.set_ylim(-0.05, 1.12)
    ax_sig.set_ylabel("normalised signal", fontsize=9)
    ax_sig.legend(loc="upper left", fontsize=8, facecolor=C["panel"],
                  edgecolor=C["grid"], labelcolor=C["text"], framealpha=0.92)

    # ── 5b. Latent trajectory — one panel per signal ───────────
    lat_cfg = [
        ("ww_signal_norm",   C["ww"],      "wastewater national", plt.cm.cool),
        ("ww_spatial_norm",  "#00e5ff",    "wastewater spatial",  plt.cm.winter),
        ("cases_pm_norm",    C["cases"],   "cases/million",       plt.cm.autumn),
        ("deaths_pm_norm",   C["excess"],  "deaths/million",      plt.cm.Reds),
        ("excess_mort_norm", "#c77dff",    "excess mort.",        plt.cm.Purples),
    ]

    for col, color, label, cmap in lat_cfg:
        ax = ax_lat[col]
        ax.set_title(f"Latent Trajectory  ({label})",
                     color=C["text"], fontsize=9, fontweight="bold", pad=6, loc="left")

        if col not in merged.columns:
            ax.text(0.5, 0.5, "no data", color=C["subtext"],
                    ha="center", va="center", transform=ax.transAxes)
            continue

        s = merged[col].values
        lx, ly = _latent(s)
        n  = len(lx)
        nm = Normalize(0, n)

        for i in range(1, n):
            ax.plot(lx[i-1:i+1], ly[i-1:i+1],
                    color=cmap(nm(i)), lw=0.9, alpha=0.75)

        # Mark known wave onsets on the trajectory
        for wname, onset_str in KNOWN_WAVES.items():
            onset = pd.Timestamp(onset_str)
            if merged.index.min() <= onset <= merged.index.max():
                wi = merged.index.searchsorted(onset, side="left")
                if wi < n:
                    ax.plot(lx[wi], ly[wi], "o", color=C["warn"],
                            ms=4, zorder=5)
                    ax.annotate(wname, (lx[wi], ly[wi]),
                                fontsize=6, color=C["warn"],
                                xytext=(4, 4), textcoords="offset points")

        # Quality indicator in corner
        q_col = col.replace("_norm", "_quality")
        if q_col in merged.columns:
            mq = merged[q_col].mean()
            ax.text(0.98, 0.02, f"quality={mq:.2f}",
                    ha="right", va="bottom", fontsize=7,
                    color=C["score"] if mq >= 0.8 else
                          C["warn"]  if mq >= 0.5 else C["excess"],
                    transform=ax.transAxes)

        sm = ScalarMappable(cmap=cmap, norm=nm)
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, pad=0.02, fraction=0.04)
        cb.set_label("time", color=C["subtext"], fontsize=7)
        cb.ax.tick_params(labelcolor=C["subtext"], labelsize=6)
        ax.set_xlabel("latent_x", fontsize=8)
        ax.set_ylabel("latent_y", fontsize=8)

    # ── 5c. Fusion score + CSD score (full width) ──────────────
    ax_sc.set_title("Fusion Trajectory Score  s(t)  +  Critical Slowing Down",
                    color=C["text"], fontsize=11, fontweight="bold", pad=8, loc="left")

    fs = merged["fusion_score"].values
    ax_sc.fill_between(dates, fs, alpha=0.18, color=C["score"])
    ax_sc.plot(dates, fs, color=C["score"], lw=2.0, label="fusion score")
    ax_sc.plot(dates, thr_arr, color=C["warn"], lw=1.2, ls="--", alpha=0.85,
               label=f"adaptive threshold (k={ADAPT_K})")

    if "csd_score" in merged.columns:
        cs = merged["csd_score"].values
        ax_sc.plot(dates, cs, color="#c77dff", lw=1.3, alpha=0.75,
                   ls="-", label="CSD score (AR1+var)")

    for name, onset_str in KNOWN_WAVES.items():
        onset = pd.Timestamp(onset_str)
        if merged.index.min() <= onset <= merged.index.max():
            ax_sc.axvline(onset, color=C["warn"], lw=0.8, ls=":", alpha=0.5)

    ax_sc.set_ylim(-0.04, 1.08)
    ax_sc.set_ylabel("score", fontsize=9)
    ax_sc.legend(loc="upper left", fontsize=8, facecolor=C["panel"],
                 edgecolor=C["grid"], labelcolor=C["text"], framealpha=0.92)

    # ── 5d. Alert comparison — 4 channels ─────────────────────
    ax_cmp.set_title(
        "Alert Timeline  (classic  |  fusion  |  spatial  |  CSD)",
        color=C["text"], fontsize=10, fontweight="bold", pad=6, loc="left")

    for i, v in enumerate(classic_alert):
        if v: ax_cmp.axvline(dates[i], ymin=0.76, ymax=0.97,
                             color=C["cases"], lw=0.7, alpha=0.7)
    for i, v in enumerate(traj_alert):
        if v: ax_cmp.axvline(dates[i], ymin=0.51, ymax=0.72,
                             color=C["score"], lw=0.9, alpha=0.85)
    for i, v in enumerate(spatial_alert):
        if v: ax_cmp.axvline(dates[i], ymin=0.26, ymax=0.47,
                             color="#00e5ff", lw=0.9, alpha=0.85)
    for i, v in enumerate(csd_alert):
        if v: ax_cmp.axvline(dates[i], ymin=0.03, ymax=0.22,
                             color="#c77dff", lw=0.9, alpha=0.85)

    for name, onset_str in KNOWN_WAVES.items():
        onset = pd.Timestamp(onset_str)
        if merged.index.min() <= onset <= merged.index.max():
            ax_cmp.axvline(onset, color=C["warn"], lw=0.9, ls=":", alpha=0.55)
            ax_cmp.text(onset, 0.60, name, color=C["warn"], fontsize=6.5,
                        ha="center", va="center",
                        transform=ax_cmp.get_xaxis_transform())

    ax_cmp.set_yticks([0.12, 0.37, 0.62, 0.87])
    ax_cmp.set_yticklabels(["CSD (ours)", "spatial (ours)", "fusion (ours)", "classic"],
                           color=C["text"], fontsize=9)
    ax_cmp.tick_params(axis="y", length=0)
    ax_cmp.set_xlabel("Date", fontsize=9)

    # ── 5e. Results table + FPR ────────────────────────────────
    ax_tbl.set_facecolor(C["bg"])
    for sp in ax_tbl.spines.values(): sp.set_visible(False)
    ax_tbl.set_xticks([]); ax_tbl.set_yticks([])

    if not results.empty:
        available_cols = [c for c in ["wave","onset","traj","spatial","csd","classic",
                                       "adv_traj","adv_spatial","adv_csd"]
                          if c in results.columns]
        col_labels_map = {
            "wave": "Wave", "onset": "Onset",
            "traj": "Fusion", "spatial": "Spatial",
            "csd": "CSD", "classic": "Classic",
            "adv_traj": "Adv.fusion", "adv_spatial": "Adv.spatial",
            "adv_csd": "Adv.CSD",
        }
        col_labels = [col_labels_map[c] for c in available_cols]
        table_data = results[available_cols].values.tolist()
        tbl = ax_tbl.table(
            cellText=table_data, colLabels=col_labels,
            loc="upper center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(7.5)
        adv_cols = [i for i, c in enumerate(available_cols) if c.startswith("adv_")]
        for (r, c), cell in tbl.get_celld().items():
            cell.set_facecolor(C["panel"] if r > 0 else C["grid"])
            cell.set_edgecolor(C["grid"])
            base_color = C["text"]
            if r > 0:
                if available_cols[c] == "adv_traj":   base_color = C["score"]
                elif available_cols[c] == "adv_spatial": base_color = "#00e5ff"
                elif available_cols[c] == "adv_csd":  base_color = "#c77dff"
            cell.set_text_props(color=base_color)

    if fp_report:
        fpr_txt = (
            f"FPR inter-wave —  "
            f"fusion: {fp_report['fp_traj']}/{fp_report['inter_wave_weeks']} "
            f"({fp_report['fpr_traj']*100:.1f}%)   |   "
            f"spatial: {fp_report['fp_spatial']}/{fp_report['inter_wave_weeks']} "
            f"({fp_report['fpr_spatial']*100:.1f}%)   |   "
            f"CSD: {fp_report.get('fp_csd', '?')}/{fp_report['inter_wave_weeks']} "
            f"({fp_report.get('fpr_csd', 0)*100:.1f}%)   |   "
            f"classic: {fp_report['fp_classic']}/{fp_report['inter_wave_weeks']} "
            f"({fp_report['fpr_classic']*100:.1f}%)"
        )
        ax_tbl.text(0.5, 0.05, fpr_txt, ha="center", va="bottom",
                    color=C["subtext"], fontsize=7.5,
                    transform=ax_tbl.transAxes)

    # ── Header ─────────────────────────────────────────────────
    tag = "[SYNTHETIC DATA]" if mode == "synthetic" else "[REAL DATA]"
    fig.text(0.50, 0.968,
             f"PANDEMIC EARLY WARNING  ·  Multi-Signal Validation  {tag}",
             ha="center", va="top", color=C["text"],
             fontsize=14, fontweight="bold")
    fig.text(0.50, 0.950,
             "NWSS wastewater  +  OWID cases / deaths / excess mortality  →  per-signal latent trajectories  →  fusion score",
             ha="center", va="top", color=C["subtext"], fontsize=9, style="italic")

    out = f"validation_{mode}.png"
    plt.savefig(out, dpi=150, facecolor=C["bg"], bbox_inches="tight")
    print(f"\n✓  saved {out}")
    return out



# =============================================================
# FLU VALIDATION MODULE
# Validates the same engine on influenza data (ILINet + FluNet)
# completely independently of the COVID pipeline.
# Demonstrates pathogen-agnostic architecture.
# =============================================================

# Delphi Epidata API — ILINet (weighted ILI %, national + state)
# Free, no auth required, weekly from 1997 to present.
DELPHI_NAT_URL   = "https://api.delphi.cmu.edu/epidata/fluview/?regions=nat&epiweeks={epiweeks}"
DELPHI_STATE_URL = "https://api.delphi.cmu.edu/epidata/fluview/?regions={states}&epiweeks={epiweeks}"

# FluNet USA confirmed detections — WHO surveillance outputs
# Direct CSV export (USA filter applied server-side via query param)
FLUNET_URL = (
    "https://frontdoor-l4uikgap6gz3m.azurefd.net/AFRO/FluNetInteractiveReport?"
    "areaname=United+States+of+America&outtype=0"
)
FLUNET_FALLBACK = "https://raw.githubusercontent.com/owid/covid-19-data/master/scripts/grapher/influenza.csv"

ILINET_CACHE  = "ilinet_nat_2008.json"   # from 2008-W40 to capture H1N1 2009
FLUNET_CACHE  = "flunet_raw.csv"   # WHO FluNet full dataset CSV (manual download)

# US state codes for Delphi API (all 50 states + DC)
_US_STATES = [
    "al","ak","az","ar","ca","co","ct","de","fl","ga",
    "hi","id","il","in","ia","ks","ky","la","me","md",
    "ma","mi","mn","ms","mo","mt","ne","nv","nh","nj",
    "nm","ny","nc","nd","oh","ok","or","pa","ri","sc",
    "sd","tn","tx","ut","vt","va","wa","wv","wi","wy","dc",
]

# Known influenza wave onsets — USA.
# Definition: first week when ILI % exceeds the CDC national baseline for
# at least 2 consecutive weeks. This is the CDC's own epidemiological onset
# standard, documented in FluView weekly archives and season summaries.
# Source for each date:
#   H1N1 2009:    pandemic spring wave — CDC MMWR, ILI above baseline week 16
#   H3N2 2014-15: CDC summary "activity began increasing in November" — week 46
#   H3N2 2017-18: FluView week 47 (ending Nov 25): "ILI above baseline for first time this season"
#   B/Vic 2019:   FluView week 46 (ending Nov 16): ILI above baseline ~week 45-46, 2+ weeks
#   H3N2 2022-23: exceptionally early — FluView Oct 22: ILI above 2.5% baseline (week 42)
FLU_WAVES = {
    "H1N1 pandemic" : "2009-04-19",   # week 16 — ILI above pandemic baseline
    "H3N2 2014-15"  : "2014-11-16",   # week 46 — first above 2.0% baseline, sustained
    "H3N2 2017-18"  : "2017-11-19",   # week 47 — first above 2.2% baseline this season
    "B/Victoria 19" : "2019-11-03",   # week 45 — first above 2.4% baseline for 2+ weeks
    "H3N2 2022-23"  : "2022-10-16",   # week 42 — exceptionally early, above 2.5% baseline
}

# Weeks after onset to search for detection (flu rises slower than COVID)
FLU_SEARCH_WEEKS = 18


def _epiweek_range(start="200940", end=None):
    """Return epiweek range string for Delphi API."""
    if end is None:
        import datetime
        now = datetime.date.today()
        # approximate current epiweek
        ew = now.isocalendar()
        end = f"{ew[0]}{ew[1]:02d}"
    return f"{start}-{end}"


def load_ilinet(refresh=False):
    """
    Fetch ILINet weighted ILI % from Delphi Epidata API.
    Returns dict:
      "national"  : weekly wILI national series
      "by_state"  : wide DataFrame (week x state)
      "spatial"   : fraction of states trending up + n_states_reporting
    """
    import json, os, requests

    ew_range = _epiweek_range("200840")  # start from 2008-W40 to capture H1N1 2009

    import datetime as _dt

    def _ew_to_date(ew_str):
        """
        Convert CDC MMWR epiweek (YYYYWW) to the start-of-week date (Sunday).
        CDC weeks start on Sunday. Week 1 contains Jan 4.
        Handles week 53 correctly via pure arithmetic.
        """
        year, week = int(ew_str[:4]), int(ew_str[4:])
        jan4 = _dt.date(year, 1, 4)
        # Days to subtract from Jan 4 to reach the preceding Sunday
        days_back = (jan4.weekday() + 1) % 7   # Mon=0..Sun=6 → Sun offset
        week1_sun = jan4 - _dt.timedelta(days=days_back)
        return pd.Timestamp(week1_sun + _dt.timedelta(weeks=week - 1))

    # ── National ──────────────────────────────────────────────
    if os.path.exists(ILINET_CACHE) and not refresh:
        print(f"  [ILINet] using cache: {ILINET_CACHE}")
        with open(ILINET_CACHE) as f:
            nat_data = json.load(f)
    else:
        url = DELPHI_NAT_URL.format(epiweeks=ew_range)
        print(f"  [ILINet] fetching national from Delphi API...")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        nat_data = r.json()
        with open(ILINET_CACHE, "w") as f:
            json.dump(nat_data, f)

    rows = nat_data.get("epidata", [])
    if not rows:
        raise ValueError("ILINet national: empty response from Delphi API")

    nat_raw = pd.DataFrame(rows)
    print(f"  [ILINet] API columns: {list(nat_raw.columns)}")

    # Select flexibly — wili is the key signal, others optional
    keep = [c for c in ["epiweek", "wili", "ili", "num_of_providers"]
            if c in nat_raw.columns]
    nat_df = nat_raw[keep].copy()
    nat_df["week"] = nat_df["epiweek"].astype(str).apply(_ew_to_date)

    # Use wili if available, fall back to ili
    val_col = "wili" if "wili" in nat_df.columns else "ili"
    nat_df = nat_df.sort_values("week").rename(columns={val_col: "ili_signal"})
    nat_df["ili_signal"] = pd.to_numeric(nat_df["ili_signal"], errors="coerce")
    print(f"  [ILINet] national: {len(nat_df)} weeks  "
          f"({nat_df['week'].min().date()} -> {nat_df['week'].max().date()})")

    # ── State-level for spatial score ─────────────────────────
    state_cache = "ilinet_states.json"
    if os.path.exists(state_cache) and not refresh:
        print(f"  [ILINet] using state cache: {state_cache}")
        with open(state_cache) as f:
            state_data = json.load(f)
    else:
        print(f"  [ILINet] fetching {len(_US_STATES)} states from Delphi API...")
        url = DELPHI_STATE_URL.format(
            states=",".join(_US_STATES), epiweeks=ew_range)
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        state_data = r.json()
        with open(state_cache, "w") as f:
            json.dump(state_data, f)

    st_rows = state_data.get("epidata", [])
    if st_rows:
        st_df = pd.DataFrame(st_rows)[["epiweek", "region", "ili"]].copy()
        st_df["week"] = st_df["epiweek"].astype(str).apply(_ew_to_date)
        st_df["ili"] = pd.to_numeric(st_df["ili"], errors="coerce")
        by_state = (
            st_df.groupby(["week", "region"])["ili"]
            .mean()
            .unstack("region")
            .sort_index()
        )
        print(f"  [ILINet] states: {by_state.shape[1]} regions  "
              f"{list(by_state.columns[:5])}...")

        # Spatial score: fraction of states trending up (lag=4 weeks)
        LAG = 4
        trending = (by_state > by_state.shift(LAG)).astype(float)
        reporting = by_state.notna().sum(axis=1)
        spatial = trending.sum(axis=1) / reporting.clip(lower=1)
        spatial = spatial.where(reporting >= 10, np.nan)

        spatial_df = pd.DataFrame({
            "week"               : spatial.index,
            "ili_spatial"        : spatial.values,
            "n_states_reporting" : reporting.values,
        }).reset_index(drop=True)
    else:
        print("  [ILINet] state data unavailable")
        spatial_df = None
        by_state   = None

    return {
        "national" : nat_df[["week", "ili_signal"]],
        "by_state" : by_state,
        "spatial"  : spatial_df,
    }


def load_flunet_usa(refresh=False):
    """
    Load WHO FluNet confirmed influenza detections for USA.

    File: flunet_raw.csv  (rename VIW_FNT.csv → flunet_raw.csv in project folder)
    Download from: https://www.who.int/teams/global-influenza-programme/
                   surveillance-and-monitoring/influenza-surveillance-outputs

    VIW_FNT.csv key columns:
      ISO2               — 'US' for United States
      ISO_WEEKSTARTDATE  — week start date
      INF_ALL            — total confirmed influenza (A+B)
      INF_A, INF_B       — by type (informational)
    """
    import os

    if not os.path.exists(FLUNET_CACHE):
        print(f"  [FluNet] {FLUNET_CACHE} not found.")
        print(f"  [FluNet] Rename VIW_FNT.csv to '{FLUNET_CACHE}' in project folder → skipping")
        return None

    print(f"  [FluNet] loading {FLUNET_CACHE}...")
    try:
        df = pd.read_csv(FLUNET_CACHE, low_memory=False, encoding_errors="replace")
    except Exception as e:
        print(f"  [FluNet] read error: {e} → skipping")
        return None

    print(f"  [FluNet] raw shape: {df.shape}")

    # ── Filter USA ────────────────────────────────────────────────────
    # VIW_FNT uses ISO2 = 'US'
    if "ISO2" in df.columns:
        usa = df[df["ISO2"].astype(str).str.upper() == "US"].copy()
    elif "COUNTRY_CODE" in df.columns:
        usa = df[df["COUNTRY_CODE"].astype(str).str.upper() == "USA"].copy()
    else:
        loc = next((c for c in df.columns if "country" in c.lower()), None)
        usa = df[df[loc].str.contains("United States", na=False)] if loc else df

    if len(usa) == 0:
        print("  [FluNet] no USA rows found → skipping")
        return None
    print(f"  [FluNet] USA rows: {len(usa)}")

    # ── Date ──────────────────────────────────────────────────────────
    date_col = "ISO_WEEKSTARTDATE" if "ISO_WEEKSTARTDATE" in usa.columns \
               else next((c for c in usa.columns if "date" in c.lower()), None)
    if date_col is None:
        print("  [FluNet] no date column → skipping"); return None

    usa[date_col] = pd.to_datetime(usa[date_col], errors="coerce")
    usa = usa.dropna(subset=[date_col]).sort_values(date_col)

    # ── Value: INF_ALL preferred, fallback INF_A+INF_B ───────────────
    if "INF_ALL" in usa.columns:
        val_col = "INF_ALL"
    elif "INF_A" in usa.columns and "INF_B" in usa.columns:
        usa["INF_ALL"] = (pd.to_numeric(usa["INF_A"], errors="coerce").fillna(0) +
                          pd.to_numeric(usa["INF_B"], errors="coerce").fillna(0))
        val_col = "INF_ALL"
    else:
        print("  [FluNet] no usable value column → skipping"); return None

    print(f"  [FluNet] using column: {val_col}")
    usa[val_col] = pd.to_numeric(usa[val_col], errors="coerce")

    # Keep FluNet dates as-is (ISO Monday-start).
    # Alignment with ILINet (CDC Sunday-start) is handled in align_flu()
    # via merge_asof with a 7-day tolerance.
    usa["week"] = usa[date_col].dt.normalize()

    weekly = (
        usa.groupby("week")[val_col]
        .sum()
        .reset_index()
        .rename(columns={val_col: "flu_confirmed"})
        .sort_values("week")
    )
    # 0 = no data submitted (not zero cases) — treat as missing
    weekly["flu_confirmed"] = weekly["flu_confirmed"].replace(0, np.nan)
    valid = weekly["flu_confirmed"].notna().sum()

    print(f"  [FluNet] USA weekly: {len(weekly)} weeks  "
          f"({weekly['week'].min().date()} → {weekly['week'].max().date()})  "
          f"non-zero: {valid}")
    return weekly


def align_flu(ilinet_data, flunet=None):
    """
    Align ILINet + optional FluNet into a merged DataFrame.
    Same quality-layer logic as align() for COVID data.
    """
    nat     = ilinet_data["national"]
    spatial = ilinet_data["spatial"]

    merged = nat.copy().set_index("week")
    if spatial is not None:
        sp = spatial.set_index("week")
        merged = merged.join(sp[["ili_spatial", "n_states_reporting"]], how="left")
        merged["n_states_reporting"] = merged.get(
            "n_states_reporting", pd.Series(0, index=merged.index)).fillna(0).astype(int)
    else:
        merged["ili_spatial"]        = np.nan
        merged["n_states_reporting"] = 0

    if flunet is not None:
        # FluNet uses ISO Monday-start weeks; ILINet uses CDC Sunday-start epiweeks.
        # The same epidemiological week can differ by up to 6 days.
        # Use merge_asof with a 7-day tolerance to match nearest week robustly.
        ili_df  = merged.reset_index().rename(columns={"week": "week"})
        flu_df  = flunet[["week", "flu_confirmed"]].copy()

        # Ensure identical datetime dtype to avoid pandas merge dtype error
        ili_df["week"] = pd.to_datetime(ili_df["week"]).astype("datetime64[us]")
        flu_df["week"] = pd.to_datetime(flu_df["week"]).astype("datetime64[us]")

        matched = pd.merge_asof(
            ili_df[["week"]].sort_values("week"),
            flu_df.sort_values("week"),
            on="week",
            tolerance=pd.Timedelta("7 days"),
            direction="nearest",
        ).set_index("week")["flu_confirmed"]

        # Re-index to original merged index
        merged["flu_confirmed"] = matched.values
    else:
        merged["flu_confirmed"] = np.nan

    signals = ["ili_signal", "ili_spatial", "flu_confirmed"]
    print(f"\n  [FLU ALIGN] {len(merged)} weeks  "
          f"({merged.index.min().date()} -> {merged.index.max().date()})")

    for s in signals:
        if s not in merged.columns:
            merged[s] = np.nan
        raw     = merged[s].copy()
        present = raw.notna().astype(float)
        merged[f"{s}_present"] = present
        quality = present.rolling(12, min_periods=1).mean()
        merged[f"{s}_quality"] = quality
        n_real  = int(present.sum())
        pct     = 100 * n_real / len(raw)
        status  = "OK" if pct >= 80 else ("WARN" if pct >= 50 else "POOR")
        print(f"  [FLU ALIGN] {s:<20} {n_real:>4}/{len(raw)} "
              f"({pct:5.1f}%)  [{status}]")
        merged[s] = raw.interpolate("linear").ffill().bfill()
        if s == "flu_confirmed":
            merged[s] = merged[s].clip(lower=0)

    # Normalise
    for s in ["ili_signal", "flu_confirmed"]:
        if s in merged.columns:
            col = merged[s]
            p95 = col.rolling(104, min_periods=8).quantile(0.95)
            p95 = p95.ffill().bfill().replace(0, np.nan).ffill().bfill()
            merged[f"{s}_norm"] = (col / p95).clip(0, 1)

    # Spatial: already 0-1 by construction
    if "ili_spatial" in merged.columns:
        merged["ili_spatial_norm"] = merged["ili_spatial"].clip(0, 1)

    return merged


# Flu-specific SIGNAL_CONFIG (same engine, different parameters)
FLU_SIGNAL_CONFIG = {
    "ili_signal_norm"   : dict(wl=3,  wr=3,  smooth_alpha=0.40, weight=0.60),
    "flu_confirmed_norm": dict(wl=4,  wr=4,  smooth_alpha=0.50, weight=0.40),
}
FLU_SPATIAL_CONFIG = dict(wl=2, wr=2, smooth_alpha=0.35)


def compute_flu_trajectory(merged):
    """Run the trajectory engine on flu signals."""
    individual_scores = {}
    for col, cfg in FLU_SIGNAL_CONFIG.items():
        if col not in merged.columns:
            print(f"  [FLU ENGINE] not found, skipping: {col}")
            continue
        s  = merged[col].values
        sc = _score(s, wl=cfg["wl"], wr=cfg["wr"])
        sc = _smooth(sc, alpha=cfg["smooth_alpha"])
        individual_scores[col] = sc
        merged[f"score_{col}"] = sc
        print(f"  [FLU ENGINE] scored {col:<25}  wl={cfg['wl']} wr={cfg['wr']}  "
              f"w={cfg['weight']}")

    fusion    = np.zeros(len(merged))
    total_eff = np.zeros(len(merged))
    for col, sc in individual_scores.items():
        base_w = FLU_SIGNAL_CONFIG[col]["weight"]
        q_col  = col.replace("_norm", "_quality")
        # Per-week quality — not global mean.
        # flu_confirmed is 0% when FluNet unavailable: those weeks get zero weight,
        # not a globally reduced weight applied to all 914 weeks.
        if q_col in merged.columns:
            q_vec = merged[q_col].values
        else:
            q_vec = np.ones(len(merged))
        eff_w      = base_w * q_vec
        fusion    += eff_w * sc
        total_eff += eff_w

    safe_total = np.where(total_eff > 0, total_eff, 1.0)
    fusion /= safe_total
    merged["fusion_score"] = fusion

    # Spatial channel
    if "ili_spatial_norm" in merged.columns:
        sp = merged["ili_spatial_norm"].values
        sc = _score(sp, wl=FLU_SPATIAL_CONFIG["wl"], wr=FLU_SPATIAL_CONFIG["wr"])
        sc = _smooth(sc, alpha=FLU_SPATIAL_CONFIG["smooth_alpha"])
        merged["spatial_score"] = sc
        print(f"  [FLU ENGINE] spatial_score  [independent channel]")

    # CSD channel — same function as COVID, applied to ili_signal_norm
    # ILI is a smoother, slower signal than NWSS — theoretically better for CSD.
    # Burn-in: 52 weeks. Result reported honestly regardless of sign.
    if "ili_signal_norm" in merged.columns:
        csd = _csd_score(merged["ili_signal_norm"].values)
        merged["csd_score"] = csd
        print(f"  [FLU ENGINE] csd_score  AR1+variance on ili_signal  "
              f"[independent channel, win={CSD_WIN}w]")
    else:
        merged["csd_score"] = np.nan

    return merged, individual_scores


def evaluate_flu(merged, persist=3):
    """Evaluate detector against known flu wave onsets."""
    scores = merged["fusion_score"].values
    dates  = merged.index
    n      = len(scores)

    thr_arr    = _adaptive_threshold(scores, k=1.5)  # lower k for flu — high seasonal variance
    ww_quality = merged.get("ili_signal_quality",
                            pd.Series(1.0, index=merged.index)).values
    pc         = 0
    traj_alert = np.zeros(n)
    for i in range(n):
        above = scores[i] > thr_arr[i] and ww_quality[i] >= 0.50
        pc    = pc + 1 if above else 0
        if pc >= persist:
            traj_alert[i] = 1

    # Spatial alert — ILINet covers all states, so use n_states >= 20
    sp_scores = merged["spatial_score"].values if "spatial_score" in merged.columns \
                else np.zeros(n)
    n_nodes   = merged["n_states_reporting"].values if "n_states_reporting" in merged.columns \
                else np.full(n, 51)
    sp_thr    = _adaptive_threshold(sp_scores, k=1.5)
    pc_sp     = 0
    spatial_alert = np.zeros(n)
    for i in range(n):
        above_sp = sp_scores[i] > sp_thr[i] and n_nodes[i] >= 20
        pc_sp    = pc_sp + 1 if above_sp else 0
        if pc_sp >= persist:
            spatial_alert[i] = 1

    # CSD alert — uses ili_signal quality gate (same as fusion)
    csd_scores = merged["csd_score"].values if "csd_score" in merged.columns \
                 else np.zeros(n)
    csd_thr  = _adaptive_threshold(csd_scores, k=1.5)
    pc_csd   = 0
    csd_alert = np.zeros(n)
    for i in range(n):
        csd_ok = csd_scores[i] > csd_thr[i] and ww_quality[i] >= 0.50
        pc_csd = pc_csd + 1 if csd_ok else 0
        if pc_csd >= persist:
            csd_alert[i] = 1

    # Classic: z-score on ili_signal_norm
    cas = merged["ili_signal_norm"].values if "ili_signal_norm" in merged.columns \
          else np.zeros(n)
    classic_alert = np.zeros(n)
    for i in range(8, n):
        w = cas[i-8:i]
        z = (cas[i] - w.mean()) / (w.std() + 1e-9)
        if z > 2.5:
            classic_alert[i] = 1

    # Rigidity alert — fires when regional decoupling from national field
    # exceeds adaptive threshold for 2+ consecutive weeks.
    # Definition: rigidity = 1 - corr(region, national, window=12w)
    # High rigidity means a region is dynamically autonomous from the national
    # signal — the earliest structural warning before case counts change.
    # We use rigidity_national (mean across states) for the alert.
    rig_scores = merged["rigidity_national"].values \
                 if "rigidity_national" in merged.columns else np.full(n, np.nan)
    rig_valid = ~np.isnan(rig_scores)
    rig_filled = np.where(rig_valid, rig_scores, 0.0)
    rig_thr = _adaptive_threshold(rig_filled, k=1.5)
    pc_rig = 0
    rigidity_alert = np.zeros(n)
    for i in range(n):
        if rig_valid[i] and rig_filled[i] > rig_thr[i]:
            pc_rig += 1
        else:
            pc_rig = 0
        if pc_rig >= 2:   # 2 weeks sustained (faster than fusion — early signal)
            rigidity_alert[i] = 1

    # Search window: 8 weeks before onset to 18 weeks after.
    # This reveals if the alert was already firing before the onset (early detection)
    # or only after (late detection). Using only post-onset was masking
    # alerts that had already triggered, making them appear to fire exactly at onset.
    SEARCH_BEFORE = 8
    results = []
    for name, onset_str in FLU_WAVES.items():
        onset = pd.Timestamp(onset_str)
        if onset < dates.min() or onset > dates.max():
            continue
        idx_onset = dates.searchsorted(onset, side="left")
        idx0 = max(0, idx_onset - SEARCH_BEFORE)
        idx1 = min(idx_onset + FLU_SEARCH_WEEKS, n)

        th  = np.where(traj_alert[idx0:idx1])[0]
        sh  = np.where(spatial_alert[idx0:idx1])[0]
        cdh = np.where(csd_alert[idx0:idx1])[0]
        rh  = np.where(rigidity_alert[idx0:idx1])[0]
        ch  = np.where(classic_alert[idx0:idx1])[0]

        t_t  = dates[idx0 + th[0]]  if len(th)  else None
        t_s  = dates[idx0 + sh[0]]  if len(sh)  else None
        t_cd = dates[idx0 + cdh[0]] if len(cdh) else None
        t_r  = dates[idx0 + rh[0]]  if len(rh)  else None
        t_c  = dates[idx0 + ch[0]]  if len(ch)  else None

        adv_t  = int((t_c - t_t).days / 7)  if (t_t  and t_c) else None
        adv_s  = int((t_c - t_s).days / 7)  if (t_s  and t_c) else None
        adv_cd = int((t_c - t_cd).days / 7) if (t_cd and t_c) else None
        adv_r  = int((t_c - t_r).days / 7)  if (t_r  and t_c) else None
        results.append({
            "wave"       : name,
            "onset"      : onset.date(),
            "traj"       : t_t.date()  if t_t  else "—",
            "spatial"    : t_s.date()  if t_s  else "—",
            "csd"        : t_cd.date() if t_cd else "—",
            "rigidity"   : t_r.date()  if t_r  else "—",
            "classic"    : t_c.date()  if t_c  else "—",
            "adv_traj"   : adv_t  if adv_t  is not None else "—",
            "adv_spatial": adv_s  if adv_s  is not None else "—",
            "adv_csd"    : adv_cd if adv_cd is not None else "—",
            "adv_rig"    : adv_r  if adv_r  is not None else "—",
        })

    # FPR — exclude burn-in and pre-onset search windows from inter-wave
    burn_in = np.zeros(n, dtype=bool)
    burn_in[:ADAPT_WIN] = True

    in_wave = np.zeros(n, dtype=bool)
    for onset_str in FLU_WAVES.values():
        onset = pd.Timestamp(onset_str)
        if onset < dates.min() or onset > dates.max():
            continue
        idx_onset = dates.searchsorted(onset, side="left")
        idx0 = max(0, idx_onset - SEARCH_BEFORE)
        idx1 = min(idx_onset + FLU_SEARCH_WEEKS, n)
        in_wave[idx0:idx1] = True
    inter = ~in_wave & ~burn_in

    fp_report = {
        "inter_wave_weeks": int(inter.sum()),
        "fp_traj"         : int(traj_alert[inter].sum()),
        "fp_spatial"      : int(spatial_alert[inter].sum()),
        "fp_csd"          : int(csd_alert[inter].sum()),
        "fp_rigidity"     : int(rigidity_alert[inter].sum()),
        "fp_classic"      : int(classic_alert[inter].sum()),
        "fpr_traj"        : round(traj_alert[inter].mean(), 3),
        "fpr_spatial"     : round(spatial_alert[inter].mean(), 3),
        "fpr_csd"         : round(csd_alert[inter].mean(), 3),
        "fpr_rigidity"    : round(rigidity_alert[inter].mean(), 3),
        "fpr_classic"     : round(classic_alert[inter].mean(), 3),
    }
    return (pd.DataFrame(results), traj_alert, spatial_alert, csd_alert,
            rigidity_alert, classic_alert, thr_arr, fp_report)


def plot_flu(merged, results, traj_alert, spatial_alert, csd_alert,
             rigidity_alert, classic_alert, thr_arr, fp_report):
    """Visualisation for flu validation run."""
    idx = merged.index
    dates = pd.DatetimeIndex(idx)

    fig = plt.figure(figsize=(20, 16), facecolor=C["bg"])
    gs  = gridspec.GridSpec(4, 2, figure=fig,
                            left=0.06, right=0.97,
                            top=0.92, bottom=0.05,
                            hspace=0.50, wspace=0.28,
                            height_ratios=[1.3, 0.9, 0.85, 0.45])

    ax_sig = fig.add_subplot(gs[0, :])
    ax_lat_ili = fig.add_subplot(gs[1, 0])
    ax_lat_sp  = fig.add_subplot(gs[1, 1])
    ax_sc  = fig.add_subplot(gs[2, :])
    ax_tbl = fig.add_subplot(gs[3, :])

    def _style(ax):
        ax.set_facecolor(C["panel"])
        for sp in ax.spines.values(): sp.set_edgecolor(C["grid"])
        ax.tick_params(colors=C["subtext"], labelsize=8)
        ax.grid(color=C["grid"], lw=0.5, alpha=0.6)

    for ax in [ax_sig, ax_lat_ili, ax_lat_sp, ax_sc]: _style(ax)

    # ── Signals ────────────────────────────────────────────────
    ax_sig.set_title("ILINet + FluNet — Normalised Signals",
                     color=C["text"], fontsize=11, fontweight="bold", loc="left")
    sig_map = [
        ("ili_signal_norm",    C["ww"],    "wILI national (ILINet)", 2.0),
        ("ili_spatial_norm",   ACCENT,     "ILI spatial spread",     1.4),
        ("flu_confirmed_norm", C["excess"],"flu confirmed (FluNet)",  1.0),
    ]
    for col, color, label, lw in sig_map:
        if col in merged.columns:
            ax_sig.plot(dates, merged[col].values,
                        color=color, lw=lw, alpha=0.85, label=label)
    for name, onset_str in FLU_WAVES.items():
        onset = pd.Timestamp(onset_str)
        if dates.min() <= onset <= dates.max():
            ax_sig.axvline(onset, color=C["warn"], lw=0.8, ls=":", alpha=0.6)
            ax_sig.text(onset, 0.98, name, color=C["warn"],
                        fontsize=6.5, ha="center", va="top",
                        transform=ax_sig.get_xaxis_transform())
    ax_sig.set_ylim(-0.05, 1.12)
    ax_sig.legend(loc="upper left", fontsize=8, facecolor=C["panel"],
                  edgecolor=C["grid"], labelcolor=C["text"])

    # ── Latent ILI ─────────────────────────────────────────────
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    for ax, col, cmap, label in [
        (ax_lat_ili, "ili_signal_norm",  plt.cm.cool,   "wILI national"),
        (ax_lat_sp,  "ili_spatial_norm", plt.cm.winter, "ILI spatial"),
    ]:
        ax.set_title(f"Latent Trajectory ({label})",
                     color=C["text"], fontsize=9, fontweight="bold", loc="left")
        if col not in merged.columns:
            continue
        s = merged[col].values
        lx, ly = _latent(s)
        n, nm  = len(lx), Normalize(0, len(lx))
        for i in range(1, n):
            ax.plot(lx[i-1:i+1], ly[i-1:i+1], color=cmap(nm(i)), lw=0.9, alpha=0.75)
        for wname, onset_str in FLU_WAVES.items():
            onset = pd.Timestamp(onset_str)
            if dates.min() <= onset <= dates.max():
                wi = dates.searchsorted(onset, side="left")
                if wi < n:
                    ax.plot(lx[wi], ly[wi], "o", color=C["warn"], ms=4, zorder=5)
                    ax.annotate(wname[:6], (lx[wi], ly[wi]), fontsize=5.5,
                                color=C["warn"], xytext=(3,3), textcoords="offset points")
        sm = ScalarMappable(cmap=cmap, norm=nm); sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, pad=0.02, fraction=0.04)
        cb.ax.tick_params(labelcolor=C["subtext"], labelsize=6)

    # ── Fusion score + alerts ──────────────────────────────────
    ax_sc.set_title("Fusion Score  s(t)  —  Alert Timeline",
                    color=C["text"], fontsize=10, fontweight="bold", loc="left")
    fs = merged["fusion_score"].values
    ax_sc.fill_between(dates, fs, alpha=0.15, color=C["score"])
    ax_sc.plot(dates, fs, color=C["score"], lw=1.8)
    ax_sc.plot(dates, thr_arr, color=C["warn"], lw=1.1, ls="--",
               alpha=0.8, label="adaptive threshold")
    for i, v in enumerate(classic_alert):
        if v: ax_sc.axvline(dates[i], ymin=0.90, ymax=0.99,
                            color=C["cases"], lw=0.6, alpha=0.6)
    for i, v in enumerate(traj_alert):
        if v: ax_sc.axvline(dates[i], ymin=0.78, ymax=0.88,
                            color=C["score"], lw=0.7, alpha=0.8)
    for i, v in enumerate(spatial_alert):
        if v: ax_sc.axvline(dates[i], ymin=0.66, ymax=0.76,
                            color=ACCENT, lw=0.7, alpha=0.8)
    for i, v in enumerate(csd_alert):
        if v: ax_sc.axvline(dates[i], ymin=0.54, ymax=0.64,
                            color="#c77dff", lw=0.7, alpha=0.7)
    for i, v in enumerate(rigidity_alert):
        if v: ax_sc.axvline(dates[i], ymin=0.42, ymax=0.52,
                            color="#ff9f1c", lw=0.7, alpha=0.7)
    for name, onset_str in FLU_WAVES.items():
        onset = pd.Timestamp(onset_str)
        if dates.min() <= onset <= dates.max():
            ax_sc.axvline(onset, color=C["warn"], lw=0.9, ls=":", alpha=0.5)
    ax_sc.set_ylim(-0.04, 1.08)
    ax_sc.legend(loc="upper left", fontsize=8, facecolor=C["panel"],
                 edgecolor=C["grid"], labelcolor=C["text"])

    # ── Table ──────────────────────────────────────────────────
    ax_tbl.set_facecolor(C["bg"])
    for sp in ax_tbl.spines.values(): sp.set_visible(False)
    ax_tbl.set_xticks([]); ax_tbl.set_yticks([])
    if not results.empty:
        col_labels = ["Wave","Onset","Fusion","Spatial","CSD","Rigidity",
                      "Classic","Adv.fusion","Adv.rig"]
        tbl = ax_tbl.table(
            cellText=results[["wave","onset","traj","spatial","csd","rigidity",
                               "classic","adv_traj","adv_rig"]].values.tolist(),
            colLabels=col_labels, loc="upper center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(8)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_facecolor(C["panel"] if r > 0 else C["grid"])
            cell.set_edgecolor(C["grid"])
            cell.set_text_props(
                color=C["score"]  if (c == 7 and r > 0) else
                      "#ff9f1c"   if (c == 8 and r > 0) else C["text"])
    if fp_report:
        ax_tbl.text(0.5, 0.05,
            f"FPR inter-wave — fusion: {fp_report['fp_traj']}/{fp_report['inter_wave_weeks']} "
            f"({fp_report['fpr_traj']*100:.1f}%)  |  "
            f"spatial: {fp_report['fp_spatial']}/{fp_report['inter_wave_weeks']} "
            f"({fp_report['fpr_spatial']*100:.1f}%)  |  "
            f"rigidity: {fp_report['fp_rigidity']}/{fp_report['inter_wave_weeks']} "
            f"({fp_report['fpr_rigidity']*100:.1f}%)  |  "
            f"CSD: {fp_report['fp_csd']}/{fp_report['inter_wave_weeks']} "
            f"({fp_report['fpr_csd']*100:.1f}%)  |  "
            f"classic: {fp_report['fp_classic']}/{fp_report['inter_wave_weeks']} "
            f"({fp_report['fpr_classic']*100:.1f}%)",
            ha="center", va="bottom", color=C["subtext"], fontsize=7.5,
            transform=ax_tbl.transAxes)

    fig.text(0.50, 0.965,
             "SYMPHONON PANDEMIC EWS  ·  Influenza Validation  [ILINet + FluNet]",
             ha="center", va="top", color=C["text"], fontsize=13, fontweight="bold")
    fig.text(0.50, 0.948,
             "Same engine as COVID pipeline — pathogen-agnostic architecture",
             ha="center", va="top", color=C["subtext"], fontsize=9, style="italic")

    out = "validation_flu.png"
    plt.savefig(out, dpi=150, facecolor=C["bg"], bbox_inches="tight")
    print(f"\n✓  saved {out}")
    return out


def _plot_spatial_map(by_state):
    """
    Generate state-level wastewater trajectory map from the NWSS by_state
    wide DataFrame (week x state). Saves to spatial_map_states.png.
    Integrated into the main COVID pipeline — no separate script needed.
    """
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.patches as mpatches

    STATE_GRID = {
        "AK":(0,0),"ME":(11,1),"VT":(10,1),"NH":(11,2),
        "WA":(1,1),"MT":(3,1),"ND":(5,1),"MN":(7,1),
        "OR":(1,2),"ID":(2,2),"WY":(3,2),"SD":(5,2),
        "WI":(7,2),"MI":(8,2),"NY":(10,2),"MA":(11,3),
        "CA":(1,3),"NV":(2,3),"UT":(3,3),"CO":(4,3),
        "NE":(5,3),"IA":(6,3),"IL":(7,3),"IN":(8,3),
        "OH":(9,3),"PA":(10,3),"NJ":(11,4),"CT":(11,5),
        "AZ":(2,4),"NM":(3,4),"KS":(5,4),"MO":(6,4),
        "KY":(8,4),"WV":(9,4),"VA":(10,4),"MD":(10,5),
        "DE":(11,6),"RI":(11,3),
        "TX":(4,5),"OK":(5,5),"AR":(6,5),"TN":(7,5),
        "NC":(9,5),"SC":(10,6),
        "LA":(6,6),"MS":(7,6),"AL":(8,6),"GA":(9,6),
        "FL":(9,7),"HI":(2,7),"DC":(10,5),
    }
    ABBREV_TO_NAME = {
        "AL":"Alabama","AK":"Alaska","AZ":"Arizona","AR":"Arkansas",
        "CA":"California","CO":"Colorado","CT":"Connecticut","DC":"District of Columbia",
        "DE":"Delaware","FL":"Florida","GA":"Georgia","HI":"Hawaii",
        "ID":"Idaho","IL":"Illinois","IN":"Indiana","IA":"Iowa",
        "KS":"Kansas","KY":"Kentucky","LA":"Louisiana","ME":"Maine",
        "MD":"Maryland","MA":"Massachusetts","MI":"Michigan","MN":"Minnesota",
        "MS":"Mississippi","MO":"Missouri","MT":"Montana","NE":"Nebraska",
        "NV":"Nevada","NH":"New Hampshire","NJ":"New Jersey","NM":"New Mexico",
        "NY":"New York","NC":"North Carolina","ND":"North Dakota","OH":"Ohio",
        "OK":"Oklahoma","OR":"Oregon","PA":"Pennsylvania","RI":"Rhode Island",
        "SC":"South Carolina","SD":"South Dakota","TN":"Tennessee","TX":"Texas",
        "UT":"Utah","VT":"Vermont","VA":"Virginia","WA":"Washington",
        "WV":"West Virginia","WI":"Wisconsin","WY":"Wyoming",
    }

    LAG, WIN = 4, 4
    latest_week = by_state.index[-1]
    scores, direction = {}, {}
    for state_name in by_state.columns:
        col = by_state[state_name].dropna()
        if len(col) < LAG + WIN:
            scores[state_name] = np.nan; direction[state_name] = 0; continue
        recent = col.iloc[-(WIN + LAG):]
        ups = sum(recent.iloc[i + LAG] > recent.iloc[i] for i in range(WIN))
        scores[state_name]    = ups / WIN
        direction[state_name] = ups - (WIN - ups)

    cmap_up   = LinearSegmentedColormap.from_list("up",   ["#0e1520", C["score"]])
    cmap_down = LinearSegmentedColormap.from_list("down", ["#0e1520", "#ef476f"])

    fig, ax = plt.subplots(figsize=(17, 10), facecolor=C["bg"])
    ax.set_facecolor(C["bg"]); ax.set_aspect("equal"); ax.axis("off")
    cell, pad = 1.0, 0.08

    for abbrev, (col, row) in STATE_GRID.items():
        full  = ABBREV_TO_NAME.get(abbrev, abbrev)
        score = scores.get(full, np.nan)
        direc = direction.get(full, 0)
        if np.isnan(score):
            face = C["grid"]; tcol = C["subtext"]
        elif score >= 0.5:
            face = cmap_up((score - 0.5) / 0.5 * 0.9 + 0.1); tcol = C["text"]
        else:
            face = cmap_down((0.5 - score) / 0.5 * 0.7 + 0.05); tcol = C["text"]
        x = col * (cell + pad); y = -row * (cell + pad)
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), cell, cell, boxstyle="round,pad=0.05",
            facecolor=face, edgecolor=C["grid"], linewidth=0.8, zorder=2))
        ax.text(x+cell/2, y+cell*0.60, abbrev,
                ha="center", va="center", fontsize=7.5,
                fontweight="bold", color=tcol, zorder=3)
        if not np.isnan(score):
            ax.text(x+cell/2, y+cell*0.28,
                    f"{'▲' if direc>0 else '▼'} {score:.0%}",
                    ha="center", va="center", fontsize=6, color=tcol, zorder=3)

    # Legend
    lx, ly = 13.4, -0.4
    ax.text(lx, ly, "Trajectory Score", color=C["text"], fontsize=9.5, fontweight="bold")
    for label, face in [
        ("Trending up >75%",    cmap_up(0.95)),
        ("Trending up 50-75%",  cmap_up(0.55)),
        ("Mixed 25-50%",        cmap_down(0.30)),
        ("Trending down <25%",  cmap_down(0.75)),
        ("No data",             C["grid"]),
    ]:
        ly -= 0.58
        ax.add_patch(mpatches.FancyBboxPatch(
            (lx, ly), 0.35, 0.42, boxstyle="round,pad=0.03",
            facecolor=face, edgecolor=C["grid"], linewidth=0.6, zorder=3))
        ax.text(lx+0.50, ly+0.21, label, color=C["text"], fontsize=7.5, va="center")

    valid  = {k: v for k, v in scores.items() if not np.isnan(v)}
    n_up   = sum(1 for v in valid.values() if v >= 0.5)
    ly -= 1.0
    ax.text(lx, ly,
            f"Reporting states: {len(valid)} / 51\n"
            f"Trending up:      {n_up} states\n"
            f"Trending down:    {len(valid)-n_up} states\n"
            f"National mean:    {np.mean(list(valid.values())) if valid else 0:.0%}",
            color=C["text"], fontsize=8.5, va="top", linespacing=1.8,
            bbox=dict(boxstyle="round,pad=0.45", facecolor=C["panel"],
                      edgecolor=C["grid"], linewidth=0.8))

    ax.set_xlim(-0.3, 15.0); ax.set_ylim(-9.2, 1.5)
    fig.text(0.50, 0.97,
             "SYMPHONON PANDEMIC EWS  ·  State-Level Wastewater Trajectory Map",
             ha="center", va="top", color=C["text"], fontsize=14, fontweight="bold")
    fig.text(0.50, 0.935,
             f"Trajectory score = fraction of last {WIN} weeks trending up vs {LAG}w ago  "
             f"·  week ending {latest_week.date()}",
             ha="center", va="top", color=C["subtext"], fontsize=9, style="italic")
    fig.text(0.50, 0.906,
             "Green = wastewater rising   |   Red = declining   |   Grey = insufficient data",
             ha="center", va="top", color=C["subtext"], fontsize=8.5)

    out = "spatial_map_states.png"
    plt.savefig(out, dpi=150, facecolor=C["bg"], bbox_inches="tight")
    plt.close()
    print(f"✓  saved {out}")


def compute_rigidity(by_state, national_series, window=12):
    """
    Rigidity metric from Symphonon framework (Panerati + Wilson, 2026).

    Definition:
        rigidity_region(t) = 1 - rolling_corr(signal_region(t),
                                               signal_national(t),
                                               window=W)

    Interpretation:
        rigidity ≈ 0  → region tracks national signal  (healthy coupling)
        rigidity ≈ 1  → region autonomous from national field (structural warning)

    A region can have NORMAL case levels but rigidity=1 if its dynamics
    have decoupled from the national field. This is the earliest measurable
    sign of regional autonomy before epidemic spread — distinct from a spike.

    Also computes:
        rigidity_national(t) = mean rigidity across all reporting states
        rigidity_max(t)      = max rigidity across states (most decoupled region)
        d_rigidity(t)        = week-over-week change in rigidity_national
                               (accelerating decoupling = urgent signal)

    Parameters
    ----------
    by_state      : DataFrame (week × state) of ILI% values
    national_series: Series (week) of national ILI% (index aligned to by_state)
    window        : rolling correlation window in weeks (default 12)

    Returns
    -------
    DataFrame with index=week and columns:
        rigidity_national, rigidity_max, d_rigidity
        Plus per-state rigidity columns: rigidity_<state>
    """
    if by_state is None or national_series is None:
        return None

    # Align national to by_state index
    nat = national_series.reindex(by_state.index)

    rigidity_cols = {}
    for state in by_state.columns:
        reg = by_state[state]
        # Rolling correlation: corr(region, national) over window weeks
        # min_periods=4 to handle early burn-in gracefully
        r = reg.rolling(window, min_periods=4).corr(nat)
        # rigidity = 1 - correlation, clipped to [0, 1]
        # NaN when insufficient data
        rigidity = (1 - r).clip(0, 1)
        rigidity_cols[f"rigidity_{state}"] = rigidity

    rig_df = pd.DataFrame(rigidity_cols, index=by_state.index)

    # Aggregate metrics
    rig_df["rigidity_national"] = rig_df.mean(axis=1)
    rig_df["rigidity_max"]      = rig_df.filter(
        regex="^rigidity_(?!national|max)").max(axis=1)
    rig_df["d_rigidity"]        = rig_df["rigidity_national"].diff()

    n_states = rig_df.filter(regex="^rigidity_(?!national|max|d_)").notna().sum(axis=1)
    print(f"  [RIGIDITY] computed for {by_state.shape[1]} states  "
          f"window={window}w  "
          f"({rig_df.index.min().date()} → {rig_df.index.max().date()})")
    print(f"  [RIGIDITY] mean rigidity_national: "
          f"{rig_df['rigidity_national'].mean():.3f}  "
          f"max ever: {rig_df['rigidity_max'].max():.3f}")

    return rig_df


def run_flu_pipeline(refresh=False):
    """Full flu validation pipeline: load → align → score → evaluate → plot."""
    print("\n== FLU VALIDATION (ILINet + FluNet) ========================")
    print("\n[1/4] Loading ILINet from Delphi Epidata API...")
    ilinet_data = load_ilinet(refresh=refresh)

    print("\n[2/4] Loading FluNet USA...")
    flunet = load_flunet_usa(refresh=refresh)

    print("\n[3/4] Aligning signals...")
    merged_flu = align_flu(ilinet_data, flunet)

    print("\n[3b/4] Computing rigidity (regional decoupling from national field)...")
    by_state = ilinet_data.get("by_state")
    national = ilinet_data["national"].set_index("week")["ili_signal"] \
               if ilinet_data.get("national") is not None else None
    rig_df = compute_rigidity(by_state, national, window=12)

    if rig_df is not None:
        # Merge rigidity aggregates into merged_flu for plotting and evaluation
        rig_agg = rig_df[["rigidity_national", "rigidity_max", "d_rigidity"]]
        merged_flu = merged_flu.join(rig_agg, how="left")
        print(f"  [RIGIDITY] joined to merged_flu — "
              f"{merged_flu['rigidity_national'].notna().sum()} valid weeks")

    print("\n[4/4] Computing trajectory scores...")
    merged_flu, _ = compute_flu_trajectory(merged_flu)

    print("\n-- FLU VALIDATION -------------------------------------------")
    res, t_alert, s_alert, csd_alert, rig_alert, c_alert, thr, fp = evaluate_flu(merged_flu)
    if not res.empty:
        print(res.to_string(index=False))
    print(f"\n  False positive rate (inter-wave):")
    print(f"    fusion     : {fp['fp_traj']}/{fp['inter_wave_weeks']} "
          f"({fp['fpr_traj']*100:.1f}%)")
    print(f"    spatial    : {fp['fp_spatial']}/{fp['inter_wave_weeks']} "
          f"({fp['fpr_spatial']*100:.1f}%)")
    print(f"    rigidity   : {fp['fp_rigidity']}/{fp['inter_wave_weeks']} "
          f"({fp['fpr_rigidity']*100:.1f}%)")
    print(f"    CSD        : {fp['fp_csd']}/{fp['inter_wave_weeks']} "
          f"({fp['fpr_csd']*100:.1f}%)")
    print(f"    classic    : {fp['fp_classic']}/{fp['inter_wave_weeks']} "
          f"({fp['fpr_classic']*100:.1f}%)")

    print("\n-- FLU PLOTTING ---------------------------------------------")
    plot_flu(merged_flu, res, t_alert, s_alert, csd_alert, rig_alert, c_alert, thr, fp)
    print("\nFlu validation done.")


def main():
    # Fix encoding on Windows terminals (cp1252 -> utf-8)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true",
                        help="Use synthetic data (no network needed)")
    parser.add_argument("--refresh", action="store_true",
                        help="Force re-download even if cache exists")
    parser.add_argument("--inspect", action="store_true",
                        help="Print columns and sample rows then exit (no full run)")
    parser.add_argument("--flu", action="store_true",
                        help="Run flu validation (ILINet + FluNet) instead of COVID pipeline")
    args = parser.parse_args()

    # ── Flu validation path ────────────────────────────────────
    if args.flu:
        run_flu_pipeline(refresh=args.refresh)
        return

    if args.offline:
        print("\n-- OFFLINE MODE (synthetic data) ------------------")
        merged = make_synthetic()
        mode = "synthetic"
        nwss_data = None
    else:
        print("\n-- ONLINE MODE (real data) ---------------------")
        nwss_data = None   # always defined, even if download fails
        try:
            print("\n[1/4] Downloading NWSS...")
            nwss_path = download(NWSS_URL, NWSS_CACHE, "NWSS", refresh=args.refresh)
            print("\n[2/4] Downloading OWID...")
            try:
                owid_path = download(OWID_URL, OWID_CACHE, "OWID", refresh=args.refresh)
            except Exception:
                print(f"  [OWID] primary URL failed, trying fallback...")
                owid_path = download(OWID_URL_FALLBACK, OWID_CACHE, "OWID", refresh=args.refresh)

            if args.inspect:
                inspect_csvs(nwss_path, owid_path)
                sys.exit(0)

            print("\n[3/4] Parsing and aligning...")
            nwss_data = load_nwss(nwss_path)
            owid      = load_owid(owid_path)
            merged    = align(nwss_data, owid)
            mode = "real"
        except Exception as e:
            print(f"\n  ⚠  Download failed: {e}")
            print("  → falling back to synthetic mode")
            merged = make_synthetic()
            mode = "synthetic"

    print("\n[4/4] Computing trajectory scores...")
    merged, _ = compute_trajectory(merged)

    print("\n-- VALIDATION -----------------------------------------------")
    results, traj_alert, spatial_alert, csd_alert, classic_alert, thr_arr, sp_thr, csd_thr, fp_report = evaluate(merged)
    if not results.empty:
        print(results.to_string(index=False))
    print(f"\n  False positive rate (inter-wave):")
    print(f"    fusion     : {fp_report['fp_traj']}/{fp_report['inter_wave_weeks']} "
          f"weeks  ({fp_report['fpr_traj']*100:.1f}%)")
    print(f"    spatial    : {fp_report['fp_spatial']}/{fp_report['inter_wave_weeks']} "
          f"weeks  ({fp_report['fpr_spatial']*100:.1f}%)")
    print(f"    CSD        : {fp_report.get('fp_csd',0)}/{fp_report['inter_wave_weeks']} "
          f"weeks  ({fp_report.get('fpr_csd',0)*100:.1f}%)")
    print(f"    classic    : {fp_report['fp_classic']}/{fp_report['inter_wave_weeks']} "
          f"weeks  ({fp_report['fpr_classic']*100:.1f}%)")

    print("\n-- PLOTTING -------------------------------------------------")
    plot(merged, results, traj_alert, spatial_alert, csd_alert,
         classic_alert, thr_arr, fp_report, mode=mode)

    # ── State-level spatial map ────────────────────────────────────────
    if mode == "real" and nwss_data.get("by_state") is not None:
        print("\n-- SPATIAL MAP ---------------------------------------------")
        _plot_spatial_map(nwss_data["by_state"])
    elif mode == "real":
        print("\n  [MAP] skipped — no state-level data available")

    print("\nDone.")


if __name__ == "__main__":
    main()
