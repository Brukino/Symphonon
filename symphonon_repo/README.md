# Symphonon

**A relational emergentist framework for structural dynamics in complex adaptive systems.**

*Independent research — Dario Panerati, Monte Amiata, Tuscany*  
*Status: working paper / pre-publication*

---

## Overview

Symphonon is a theoretical and computational framework that models **relational coherence** in complex adaptive systems. It operates along two distinct but connected tracks:

**Symphonon P** — a practical predictive maintenance tool validated on real wind turbine datasets. It detects structural precursors to failure using a non-Markovian order parameter that tracks latent field coherence without requiring knowledge of global system structure.

**Symphonon Ω / Gestalt** — a theoretical simulation framework modeling coupled phase fields, emergent agents (poli), and higher-order organisms (gestalt, supergestalt). It explores how relational coherence produces persistent structure from local dynamics.

---

## Epistemological Position

The framework formalises a shift from *static metric* to *dynamic observer*: we do not measure the structural state of a system — we observe its trajectory.

The core claim: identity is not a primitive. What we call an object is a compressed, self-consistent trajectory within a relational field. The observer itself is a condition of possibility of the field it navigates.

---

## Repository Structure

```
symphonon/
├── symphonon_P/           # Predictive maintenance — wind turbine validation
│   ├── symphonon_P_v4_1_local_runner.py   # Main runner (Kelmarsh/Penmanshiel)
│   ├── symphonon_probe_v17.py             # Core probe implementation
│   ├── wind_turbine_precursor.py          # Precursor detection
│   ├── wind_full_validation.py            # Three-site validation
│   ├── wind_false_alarm.py                # False alarm analysis
│   ├── wind_ablation.py                   # Ablation study
│   ├── wind_bootstrap.py                  # Bootstrap confidence intervals
│   ├── wind_multiyear.py                  # Multi-year analysis
│   ├── penmanshiel_validation.py          # Penmanshiel site
│   ├── symphonon_fv_runner.py             # Fault validation runner
│   └── symphonon_fv_fault_injection.py    # Fault injection framework
│
├── symphonon_omega/       # Field simulation — emergent dynamics
│   ├── symphonon_gestalt_v4.py            # Current simulation (pygame)
│   ├── symphonon_gestalt_v3_archive.py    # Previous version
│   ├── symphonon_v46.py                   # Core field dynamics
│   └── symphonon_variational_v16.py       # Variational formulation
│
├── docs/
│   ├── SYMPHONON_FORMAL_FRAMEWORK.md      # Formal framework document
│   ├── results.json                       # Validation results
│   └── summary.txt                        # Results summary
│
└── figures/               # Validation figures
```

---

## Symphonon P — Validated Results

Three-site validation on public datasets:

| Dataset | Source | Turbines | FA rate | Detection |
|---------|--------|----------|---------|-----------|
| Kelmarsh | Zenodo 5841834 | T01–T06 | 0.18/month | DR 0.76 |
| Penmanshiel | Zenodo 5946808 | multiple | 0.22/month | DR 0.71 |
| Nørrekær Enge | DTU figshare | 3 turbines | 0.37/month | sensor-limited |

**Key metrics (Kelmarsh, primary validation site):**
- Detection rate: **0.76** vs EWMA baseline 0.41
- False alarm rate: **0.18/month** vs threshold-based 0.91/month
- Mean advance warning: **23 days** before documented fault

**Publishability condition** (pre-stated):
```
t_peak(tension) < t_peak(o)  on ≥ 5/6 replicas — verified 6/6
```

---

## Symphonon Ω — Core Architecture

Three coupled fields, never all simultaneously observable:

```
φ(x,t)  — fast XY phase field (Kuramoto-Langevin, quasi-Markovian)
ψ(x,t)  — slow Allen-Cahn field with spatial memory (non-Markovian)
T(x,t)  — latent interface: T = |Vort_φ| · (1−Kap_ψ) · (0.5+Mu_ψ)
           never directly rendered
```

Emergent hierarchy:
```
Poli        → conditions of possibility navigating T
Gestalt     → second-order organisms from pole fusion
SuperGestalt → relational structure between coherent Gestalt
```

**Empirical findings (Gestalt v4, >15,000 dissolution events):**

1. Gestalt dissolution is purely emergent — continuous energy balance `dV/dt = gain(T_local) − decay`, no arbitrary threshold.

2. T_local on Gestalt ≈ 9.6× T_global — agents navigate toward vorticity maxima. Calibration must use T_local, not T_global.

3. SuperGestalt effect on survival: +1.7% life extension despite navigating zones with 44% lower T_local after SG entry. The relational constraint reduces navigational freedom but provides structural stabilisation — analogous to protein complex quaternary structure.

---

## Formal Framework

The formal framework is in `docs/SYMPHONON_FORMAL_FRAMEWORK.md`.

Key structures:

**Order parameter (non-Markovian):**
```
o(t+1) = (1−β)·o(t) + β·λ₂(t) + ε(t)
```

**Tension (implicit temporal derivative):**
```
tension(t) = o(t) − λ₂(t) ≈ −(1/β)·ȯ(t)
```

**Gestalt existence condition (from ROF7):**
```
∂(G) = |⟨e^{iφ}⟩_internal − ⟨e^{iφ}⟩_external| > θ_boundary
```

---

## Datasets

- **Kelmarsh**: [Zenodo 10.5281/zenodo.5841834](https://zenodo.org/record/5841834)
- **Penmanshiel**: [Zenodo 10.5281/zenodo.5946808](https://zenodo.org/record/5946808)
- **Nørrekær Enge**: DTU figshare (not Zenodo)

---

## Dependencies

**Symphonon P:**
```
numpy, pandas, scipy, sklearn, matplotlib
```

**Symphonon Ω:**
```
numpy, pygame (pip install pygame)
python3 symphonon_gestalt_v4.py
```

---

## IP

Framework deposited with SIAE OLAF (Italian copyright registration) prior to public release.

---

## Contact

Dario Panerati — independent researcher, Monte Amiata, Tuscany, Italy.

*This is pre-publication research. The framework is shared for scientific exchange, not as a finished product.*
