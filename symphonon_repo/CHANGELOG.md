# Symphonon — Changelog

## v0.2.0 — 2026-04-22

### Results

**Selection bias measurement (new empirical result, Type B):**
- T_local/T_global = 8.63 +/- 0.26 (CoV=3.0%, 95% CI [8.45, 8.81], n=8)
- Decomposition: 96.5% intrinsic (Kuramoto field structure) + 3.5% induced (perturb_phi)
- Excluded causes: navigation geometry, psi field, perturb_phi as primary driver
- Pending: random navigation test to isolate active navigation contribution

### New metrics in symphonon_gestalt_v4.py
- `rigidity` — field decoupling metric: 1 - corr(T_local, T_global)
- `d_rigidity` — rate of decoupling (EMA of derivative)
- `o_local` — per-agent order parameter (Interpretant in Peircean sense)
- Logged in [DISS] and [SG] events

### New files
- `rigidity_metrics.py` — field vs fleet rigidity (rolling z-score, no leakage)
- `rigidity_analysis.py` — Check 1-4 falsification protocol (Wilson 2026)
- `symphonon_semiosis_test.py` — PE test + 3-language convergence + Layer 3 protocol
- `geometry_experiment_full.py` — geometry effect on T ratio
- `psi_test.py` — psi field contribution test
- `pert_test.py` — perturb_phi contribution test
- `baseline_test.py` — baseline stability (n=8)
- `random_nav_test.py` — navigation strategy test (pending results)
- `state.json` — persistent dialogue state

### Framework
- `SYMPHONON_FORMAL_FRAMEWORK_v2.md` — section 13: selection bias (Type B result)
- `ROF_Symphonon_joint_note.md` — section 6: rigidity + false(true) mapping

### ROF-Peirce-Symphonon interface
- Peircean triad mapping: Firstness/Secondness/Thirdness <-> field zones
- o_local as Interpretant (non-Markovian internal state)
- T_local/T_global as Immediate/Dynamic Object ratio (measured: 8.63x)
- Inter-agent semiosis Layer 1/2/3 formalised, Layer 3 test protocol ready

---

## v0.1.0 — 2026-04-17

- Symphonon P validated: 6/7 faults, 0.22 FA/month (Kelmarsh)
- Penmanshiel zero-shot replication: 0.041 FA/month
- Gestalt v4 stable architecture
- FORMAL_FRAMEWORK_v2 with claim discipline A/B/C/D
