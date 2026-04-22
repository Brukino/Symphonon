"""
Symphonon Gestalt v4 - Random Navigation Test
=============================================
Risponde alla domanda: perturb_phi (poli che perturbano il campo phi)
e' responsabile del rapporto T_locale/T_global ~ 9.0x?

Risultati precedenti:
  - geometria navigazione: esclusa
  - campo psi (Allen-Cahn): escluso (ratio=9.063 con psi=0)

Meccanismo ipotizzato:
  I poli perturbano phi creando gradienti che generano vorticity locale.
  I gestalt navigano verso quella vorticity -> T_locale >> T_global.
  Senza perturbazioni: T distribuito uniformemente -> ratio -> 1.

Meccanismo ipotizzato:
  psi crea memoria spaziale che amplifica T nelle zone ad alta vorticity.
  I gestalt navigano verso questi massimi -> T_locale >> T_global.
  Senza psi: T uniforme -> ratio ~ 1.

Condizioni testate:
  alpha=0.00  ? psi disabilitato (PSI_ALPHA=0)
  alpha=0.01  ? psi ridotto
  alpha=0.03  ? default (valore corrente)
  alpha=0.06  ? psi aumentato

Predizione:
  ratio cresce monotonicamente con alpha
  -> psi e la causa del 9.6x

Uso:
  Metti nella stessa cartella di symphonon_gestalt_v4.py
  python psi_test.py

Output: tabella + psi_results.csv
"""

import sys, os, math, csv, types
import numpy as np
_np = np  # alias per uso dentro funzioni

# ?? Fake pygame ???????????????????????????????????????????????????????????????
fake_pygame = types.ModuleType('pygame')
fake_pygame.init = lambda: None
fake_pygame.quit = lambda: None
fake_pygame.display = types.SimpleNamespace(
    set_caption=lambda *a,**k: None,
    Info=lambda: types.SimpleNamespace(current_w=1920,current_h=1080),
    set_mode=lambda *a,**k: None,
    flip=lambda: None,
)
fake_pygame.FULLSCREEN=fake_pygame.MOUSEBUTTONDOWN=fake_pygame.MOUSEBUTTONUP=0
fake_pygame.MOUSEMOTION=fake_pygame.MOUSEWHEEL=fake_pygame.KEYDOWN=0
fake_pygame.K_SPACE=fake_pygame.K_r=fake_pygame.K_s=fake_pygame.K_d=fake_pygame.K_ESCAPE=0
fake_pygame.event=types.SimpleNamespace(get=lambda:[])
fake_pygame.mouse=types.SimpleNamespace(get_pos=lambda:(0,0))
fake_pygame.draw=types.SimpleNamespace(
    circle=lambda*a,**k:None, line=lambda*a,**k:None, lines=lambda*a,**k:None)
fake_pygame.Surface=lambda*a,**k:types.SimpleNamespace(
    blit=lambda*a,**k:None, fill=lambda*a,**k:None, set_alpha=lambda*a,**k:None)
fake_pygame.font=types.SimpleNamespace(
    SysFont=lambda*a,**k:types.SimpleNamespace(
        render=lambda*a,**k:types.SimpleNamespace(get_width=lambda:0,get_height=lambda:0)),
    init=lambda:None)
fake_pygame.time=types.SimpleNamespace(
    Clock=lambda:types.SimpleNamespace(tick=lambda x:None,get_fps=lambda:60))
sys.modules['pygame'] = fake_pygame

# ?? Parametri ?????????????????????????????????????????????????????????????????
N_FRAMES  = 4000   # frame dopo warmup — piu lungo per baseline stabile
WARMUP    = 1500   # warmup
N_RUNS    = 8      # run per condizione — piu run per CI robusto

# Ora alpha e' un moltiplicatore di Psi (0=disabilitato, 1=default)
# Non modifica PSI_ALPHA globale ma scala/azzera Psi direttamente
# Moltiplicatori di PERT_STR e GEST_PERT
# 0.0 = nessuna perturbazione, 1.0 = default
# 0.0 = navigazione casuale (random walk)
# 1.0 = navigazione standard (gradient ascent su T)
NAV_LIST = [0.0, 1.0]


# ?? Import simulazione ????????????????????????????????????????????????????????
def import_sim():
    import importlib.util
    candidates = ['symphonon_gestalt_v4.py',
                  os.path.join(os.path.dirname(__file__), 'symphonon_gestalt_v4.py')]
    for c in candidates:
        if os.path.exists(c):
            spec = importlib.util.spec_from_file_location('gestalt_v4', c)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    print("ERRORE: symphonon_gestalt_v4.py non trovato.")
    sys.exit(1)


# ?? Runner per una condizione ?????????????????????????????????????????????????
def run_nav(mod, nav_mode: float, seed: int = 42) -> dict:
    """
    Esegue un run con PSI_ALPHA specificato.
    Usa mod.step() direttamente ? nessun patching della navigazione.
    Misura solo T_locale/T_global.
    """
    np.random.seed(seed)

    # Crea simulazione e usa psi_alpha_override sull'istanza
    # Questo e piu affidabile del patch sulla costante globale:
    # il metodo field.step() legge _psi_alpha da self invece che dal modulo
    sim = mod.Simulation()
    # nav_mode=0: sostituisce la navigazione con random walk dopo ogni step
    # nav_mode=1: navigazione standard (nessuna modifica)
    # Il random walk viene applicato modificando direttamente gx,gy dei gestalt

    T_local_records  = []
    T_global_records = []

    for frame in range(N_FRAMES + WARMUP):
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            sim.step()


        # Se nav_mode=0: sovrascrive la posizione con random walk
        if nav_mode == 0.0:
            import math as _m
            for g in sim.gestalts:
                if not g.dead and g.dissolving == 0:
                    angle = _np.random.uniform(0, 2*_m.pi)
                    step  = mod.GEST_STEP
                    g.gx  = max(2, min(mod.G-2, g.gx + _m.cos(angle)*step))
                    g.gy  = max(2, min(mod.G-2, g.gy + _m.sin(angle)*step))
                    # Ricalcola T dopo spostamento casuale
                    ci_x  = int(round(max(0, min(mod.G-1, g.gx))))
                    ci_y  = int(round(max(0, min(mod.G-1, g.gy))))
                    g.T   = float(sim.field.T_field[ci_y, ci_x]) if hasattr(sim.field, 'T_field') else sim.field.T_global

        # Progresso ogni 500 frame su stderr
        if frame % 500 == 0:
            import sys as _sys
            n_g = len(sim.gestalts)
            T_g = sim.field.T_global
            T_l = sum(g.T for g in sim.gestalts)/max(1,n_g) if sim.gestalts else 0
            r_now = T_l/T_g if T_g > 0 else 0
            _sys.stderr.write(f"    frame={frame} T_glob={T_g:.4f} T_loc={T_l:.4f} ratio={r_now:.2f} gest={n_g}\n")
            _sys.stderr.flush()

        # Misura dopo warmup
        if frame >= WARMUP and sim.gestalts:
            T_locals = [g.T for g in sim.gestalts if g.T > 0]
            if T_locals and sim.field.T_global > 0:
                T_local_records.append(sum(T_locals)/len(T_locals))
                T_global_records.append(sim.field.T_global)

    if not T_local_records:
        return {'ratio': None, 'T_local': None, 'T_global': None,
                'n_gest_mean': 0}

    T_loc  = sum(T_local_records) / len(T_local_records)
    T_glob = sum(T_global_records) / len(T_global_records)
    ratio  = T_loc / T_glob if T_glob > 0 else None

    return {
        'ratio':    ratio,
        'T_local':  T_loc,
        'T_global': T_glob,
        'n_frames': len(T_local_records)
    }


# ?? Main ??????????????????????????????????????????????????????????????????????
def main():
    print("="*65)
    print("  RANDOM NAVIGATION TEST")
    print("  La navigazione attiva causa il selection bias?")
    print("="*65)
    print(f"  Frames: {N_FRAMES} + warmup {WARMUP} | Run: {N_RUNS}")
    print(f"  nav=0: random walk | nav=1: gradient ascent su T")
    print(f"  nav=0.0  -> random walk (nessun gradient ascent)")
    print(f"  nav=1.0  -> navigazione standard (gradient ascent su T)")
    print()
    print("  Caricamento simulazione...")

    mod = import_sim()
    print(f"  Caricato. G={mod.G}, GEST_STEP={mod.GEST_STEP}")

    results = []

    print()
    print(f"  {'PSI_ALPHA':>10}  {'ratio':>8}  {'T_local':>10}  "
          f"{'T_global':>10}  {'std':>7}  {'note':>20}")
    print("  " + "-"*65)

    for alpha in NAV_LIST:
        ratios   = []
        T_locs   = []
        T_globs  = []

        for run in range(N_RUNS):
            import sys as _sys
            _sys.stderr.write(f"  alpha={alpha:.2f} run={run+1}/{N_RUNS}...\n")
            _sys.stderr.flush()
            print(f"    alpha={alpha:.2f} run={run+1}/{N_RUNS}...", end=' ', flush=True)
            r = run_nav(mod, nav_mode=alpha, seed=42+run)
            if r['ratio'] is not None:
                ratios.append(r['ratio'])
                T_locs.append(r['T_local'])
                T_globs.append(r['T_global'])
                print(f"ratio={r['ratio']:.3f}")
            else:
                print("no data")

        if ratios:
            mean_r  = sum(ratios) / len(ratios)
            std_r   = (sum((x-mean_r)**2 for x in ratios)/len(ratios))**0.5
            mean_Tl = sum(T_locs) / len(T_locs)
            mean_Tg = sum(T_globs) / len(T_globs)

            # Nota interpretativa
            if alpha == 0.0:
                note = "random walk"
            elif mean_r < 2.0:
                note = "ratio basso"
            elif mean_r < 5.0:
                note = "ratio medio"
            else:
                note = "ratio alto"

            print(f"  {alpha:>10.2f}  {mean_r:>8.3f}  {mean_Tl:>10.5f}  "
                  f"{mean_Tg:>10.5f}  {std_r:>7.3f}  {note:>20}")

            results.append({
                'psi_alpha': alpha,
                'ratio_mean': mean_r,
                'ratio_std':  std_r,
                'T_local':    mean_Tl,
                'T_global':   mean_Tg
            })
        print()

    # Salva CSV
    csv_path = 'random_nav_results.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['psi_alpha','ratio_mean','ratio_std','T_local','T_global'])
        for r in results:
            w.writerow([r['psi_alpha'], r['ratio_mean'], r['ratio_std'],
                        r['T_local'], r['T_global']])
    print(f"\n  Salvato: {csv_path}")

    # Sintesi
    print()
    print("="*65)
    print("  BASELINE STABILITY RESULTS")
    print("="*65)

    r0   = next((r for r in results if r['psi_alpha'] == 0.0), None)
    r1   = next((r for r in results if r['psi_alpha'] == 1.0), None)

    import math as _m
    if r0 and r1:
        ci0 = 1.96 * r0['ratio_std'] / _m.sqrt(N_RUNS)
        ci1 = 1.96 * r1['ratio_std'] / _m.sqrt(N_RUNS)
        print(f"  Random walk (nav=0):")
        print(f"    ratio = {r0['ratio_mean']:.4f} +/- {r0['ratio_std']:.4f}")
        print(f"    95% CI: [{r0['ratio_mean']-ci0:.4f}, {r0['ratio_mean']+ci0:.4f}]")
        print()
        print(f"  Navigazione standard (nav=1):")
        print(f"    ratio = {r1['ratio_mean']:.4f} +/- {r1['ratio_std']:.4f}")
        print(f"    95% CI: [{r1['ratio_mean']-ci1:.4f}, {r1['ratio_mean']+ci1:.4f}]")
        print()
        delta = r1['ratio_mean'] - r0['ratio_mean']
        pct   = delta / max(r0['ratio_mean'], 0.001) * 100
        print(f"  Delta (nav1 - nav0) = {delta:+.4f} ({pct:+.1f}%)")
        print()
        if r0['ratio_mean'] < 2.0:
            print("  -> CONFERMATO: navigazione attiva causa il selection bias")
            print("     Random walk -> ratio ~1 (nessun bias)")
            print("     Gradient ascent -> ratio ~8.6x (bias presente)")
        elif r0['ratio_mean'] > 5.0:
            print("  -> NON CONFERMATO: ratio alto anche con random walk")
            print("     La struttura del campo genera l'aggregazione")
            print("     indipendentemente dalla strategia di navigazione")
        else:
            print(f"  -> PARZIALE: random walk ratio={r0['ratio_mean']:.2f}x")
            print("     Sia il campo che la navigazione contribuiscono")

    print("="*65 + "\n")


if __name__ == '__main__':
    main()
