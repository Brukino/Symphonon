# Symphonon P — Predictive Maintenance

Non-Markovian structural precursor detection for wind turbines.

## Quick start

```bash
# Install dependencies
pip install numpy pandas scipy scikit-learn matplotlib

# Run on Kelmarsh data (download from Zenodo 5841834)
python symphonon_P_v4_1_local_runner.py
```

## File guide

| File | Purpose |
|------|---------|
| `symphonon_P_v4_1_local_runner.py` | Main runner — reads parquet, runs detection, outputs metrics |
| `symphonon_probe_v17.py` | Core probe: o(t), tension, regime classifier |
| `wind_turbine_precursor.py` | Precursor signal construction |
| `wind_full_validation.py` | Three-site validation pipeline |
| `wind_false_alarm.py` | False alarm rate analysis |
| `wind_ablation.py` | Ablation: which components matter |
| `wind_bootstrap.py` | Bootstrap confidence intervals on metrics |
| `wind_multiyear.py` | Multi-year longitudinal analysis |
| `penmanshiel_validation.py` | Penmanshiel-specific runner |
| `symphonon_fv_runner.py` | Fault validation: retrospective analysis |
| `symphonon_fv_fault_injection.py` | Synthetic fault injection for calibration |
| `bonn_precursor.py` | Additional domain validation |
| `pandemic_demo.py` | Cross-domain demo (epidemiological data) |

## Dataset access

Kelmarsh: https://zenodo.org/record/5841834  
Penmanshiel: https://zenodo.org/record/5946808  
Nørrekær Enge: DTU figshare (search "Nørrekær Enge SCADA")
