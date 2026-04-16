# Symphonon Ω — Field Simulation

Coupled phase field simulation with emergent agents and hierarchical structures.

## Quick start

```bash
pip install numpy pygame
python symphonon_gestalt_v4.py > log.txt 2>&1
```

Controls: `R` reset · `S` shock · `+/-` zoom · mouse drag pan

## Architecture

Three coupled fields:
- `φ(x,t)` — fast XY phase field (Kuramoto-Langevin)
- `ψ(x,t)` — slow Allen-Cahn field with spatial memory  
- `T(x,t)` — latent interface, never directly rendered

Emergent hierarchy: **Poli → Gestalt → SuperGestalt**

## Log format

```
[FIELD] frame=N T=global T_gest=local noise=gain poles=N gest=N sg=N
[FUSE]  frame=N coh=X dG=Y n_poles=N gestalt_id=N
[DISS]  frame=N vita=Nf T=local T_avg=lifetime_mean T_pre=before_sg T_post=after_sg sg=id
[SG]    Formato SuperGestalt N — coh=X dist=Y
```

## Key empirical findings

- T_local on agents ≈ 9.6× T_global (agents navigate to vorticity maxima)
- Gestalt dissolution: purely emergent, continuous energy balance
- SuperGestalt entry reduces T_local by ~44% but extends life by ~1.7%
- Mechanism: relational constraint stabilises despite reduced navigational freedom

## Files

| File | Description |
|------|-------------|
| `symphonon_gestalt_v4.py` | Current version — full hierarchy with diagnostics |
| `symphonon_gestalt_v3_archive.py` | Previous version (reference) |
| `symphonon_v46.py` | Core field dynamics module |
| `symphonon_variational_v16.py` | Variational / free energy formulation |
