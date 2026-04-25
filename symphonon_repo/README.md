# Symphonon Pandemic EWS

**AI × Biosecurity Hackathon 2026 · Track: Pandemic Early Warning Systems**  
Apart Research · BlueDot Impact · Cambridge Biosecurity Hub

---

## What this is

A trajectory-based, pathogen-agnostic early warning system for epidemic onset detection.

The core claim: epidemic systems **drift before they tip**. Wastewater rises consistently for weeks before clinical cases accumulate. Geographic spread begins in a handful of states before it becomes national. This pipeline detects that directional drift — not the spike.

The same engine, validated on two pathogens without retraining:

| | COVID-19 | Influenza |
|---|---|---|
| Validation span | 321 weeks (2019–2026) | 914 weeks (2008–2026) |
| Data sources | CDC NWSS + OWID | ILINet + WHO FluNet |
| Fusion FPR | 5.6% | 12.3% |
| Spatial FPR | 1.5% | 3.1% |
| Classic z-score FPR | 9.1% | 14.4% |
| Best advance (gradual onset) | — | **+4 to +6 weeks** (H3N2 2014–15, 2017–18) |
| Engine changes required | none | k: 2.0 → 1.5 |

---

## How it works

**Score first, then fuse — never fuse raw signals.**

Each surveillance signal is scored independently by the fraction of recent weeks where the signal was higher than it was a fixed lag ago. A score near 1.0 means consistently rising. A score near 0 means consistently falling. Scores are then fused with runtime quality-adjusted weights.

Two independent alert channels:

- **Fusion** — aggregate directional momentum across all signals
- **Spatial** — fraction of US states trending up simultaneously, gated by minimum reporting nodes (≥ 20)

Both channels use an adaptive threshold computed from trailing 26-week mean and standard deviation. No fixed thresholds.

---

## Quick start

```bash
# Install dependencies
pip install numpy pandas matplotlib scipy requests

# Run COVID-19 validation (downloads ~300MB of public data on first run)
python SymPandemic_data_pipeline.py

# Run influenza validation (requires ILINet cache files — see Data section)
python SymPandemic_data_pipeline.py --flu

# Offline mode (synthetic data, no downloads)
python SymPandemic_data_pipeline.py --offline

# Use cached data (no re-download)
python SymPandemic_data_pipeline.py --no-refresh
```

Outputs: `validation_real.png`, `validation_flu.png`, `spatial_map_states.png`

---

## Data sources

All public, all freely available.

| Source | Signal | URL |
|---|---|---|
| CDC NWSS | Wastewater viral load | `data.cdc.gov/api/views/2ew6-ywp6/rows.csv?accessType=DOWNLOAD` |
| OWID | Cases, deaths, excess mortality | `covid.ourworldindata.org/data/owid-covid-data.csv` |
| Delphi Epidata | ILINet weighted ILI % | `api.delphi.cmu.edu/epidata/fluview` |
| WHO FluNet | Confirmed influenza detections | Manual download: `who.int/tools/flunet` → VIW_FNT.csv |

For the flu validation, place the FluNet file as `flunet_raw.csv` in the working directory. ILINet data is fetched automatically via the Delphi API and cached locally.

---

## Signal configuration

| Signal | Lag (wL) | Window (wR) | Smoothing (α) | Weight | Role |
|---|---|---|---|---|---|
| NWSS wastewater | 4w | 4w | 0.45 | 45% | Leading |
| Cases/million | 6w | 6w | 0.55 | 30% | Mid-lag |
| Deaths/million | 10w | 8w | 0.70 | 15% | Lagging |
| Excess mortality | 12w | 10w | 0.75 | 10% | Lagging |
| ILI (flu only) | 3w | 3w | 0.35 | 60% | Leading |
| FluNet (flu only) | 4w | 4w | 0.45 | 40% | Mid-lag |

Parameters reflect signal lag structure, not per-pathogen tuning.

---

## Where the method works and where it does not

| Scenario | Result | Reason |
|---|---|---|
| Gradual seasonal onset (flu 2014, 2017) | +4–6 weeks advance | Consistent drift detectable before threshold crossing |
| Abrupt onset with quality data (COVID BA.5) | −3 weeks vs z-score | Magnitude spikes detected faster by z-score |
| Novel spillover (H1N1 2009) | Not detected | No pre-transition drift — abrupt unprecedented emergence |
| Post-disruption baseline (flu 2022–23) | Degraded | Adaptive threshold miscalibrated after COVID endemic shift |

---

## Channels tested and not validated

Two additional channels derived from complex-systems theory were implemented and did not add value:

- **Rigidity** (1 − rolling correlation between regional and national ILI): FPR 12.2%, 0/5 onsets detected. Fires during every seasonal cycle because regional timing varies naturally — not specific to onset. Requires a per-region outlier approach rather than national mean pooling.
- **CSD (AR(1) + variance)**: FPR 19.5%, all detections post-onset. Under strong periodic forcing, AR(1) saturates at the seasonal peak, not the beginning. Not a valid onset detector for seasonal influenza.

Both are in the codebase; neither is claimed as validated.

---

## Theoretical grounding

This work is a domain transfer of the Symphonon framework (see parent repo), which encodes complex-systems properties — directional momentum, geographic diffusion, regional decoupling — as computable scores over time-series data. The framework was originally validated on industrial prognostics (wind turbines: Kelmarsh SCADA, Penmanshiel; jet engines: NASA CMAPSS).

The rigidity metric has formal grounding in the Relational Operational Framework (Panerati & Wilson, 2026), which provides a structural account of regional autonomy as a precursor to super-spreader dynamics.

---

## Reproducibility

All numerical results in the technical brief are generated by running the two commands above. No external configuration, no trained models, no proprietary data.

Version: `symphonon-pandemic-ews-v1.0.0-alpha`

---

## License

See parent repository.
