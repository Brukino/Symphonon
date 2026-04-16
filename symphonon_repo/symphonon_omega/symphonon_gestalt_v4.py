"""
SYMPHŌNŌN Ω — GESTALT v3  (pygame)
====================================
Architettura a campi accoppiati + organismi gestalt.

Tre livelli ontologici:

  φ(x,t)   Campo XY veloce — quasi-Markoviano, Kuramoto-Langevin
            Nascosto: evolve sempre, mai visualizzato.

  ψ(x,t)   Campo lento Allen-Cahn — non-Markoviano (∇²ψ porta memoria)
            Nascosto. Accoppiato a φ tramite α·R_φ.

  T(x,t)   Interfaccia latente — MAI visualizzata
            T = |Vort_φ| · (1−Kap_ψ) · (0.5+Mu_ψ)

  Poli     Agenti individuali. Navigano il gradiente di T.
            Perturbano φ localmente (genotipo guida la perturbazione).
            Vitalità ∝ T nel sito.

  Gestalt  Organismi nati dalla fusione di poli coerenti.
            Entità fisiche autonome: corpo, fase, genotipo collettivo.
            Si muovono come unità. Perturbano φ in modo coordinato.
            Fusione tra gestalt compatibili.
            Dissoluzione → rilascia poli se il campo degrada.

Camera: pan con drag, zoom con rotella.

Tasti:
  SPACE   pausa/riprendi
  R       reset
  S       SHOCK globale su φ
  D       SEED: nucleo di coerenza in centro
  ESC     esci
"""

import pygame
import numpy as np
import sys, math, time
from dataclasses import dataclass, field
from typing import List, Optional

# ── CONFIGURAZIONE ────────────────────────────────────────────────
WIN_W, WIN_H = 1100, 750
G = 80          # griglia G×G
FPS_TARGET = 60

# ── CAMPO φ ───────────────────────────────────────────────────────
KUR_K      = 0.55
NOISE_BASE = 0.9
DT         = 0.06
SQDT       = math.sqrt(DT)

# ── CAMPO ψ ───────────────────────────────────────────────────────
PSI_D      = 0.05
PSI_KAPPA  = 0.28   # decadimento più rapido
PSI_ALPHA  = 0.03   # ridotto ulteriormente: ψ non deve saturare

# ── POLI ──────────────────────────────────────────────────────────
N_START    = 18
N_MIN      = 5
N_SOFT     = 18   # capacità portante ridotta
MAX_STEP   = 1.2             # aumentato per movimento visibile
MIN_D      = 7        # distanza minima in unità griglia
VIT_DECAY  = 0.0014  # mortalità aumentata per tenere N sotto controllo
VIT_BASE   = 0.0010
VIT_GAIN   = 0.12
VIT_DEATH  = 0.05
VIT_REPRO  = 0.55
REPRO_P    = 0.004   # riproduzione più lenta
AGE_MAX    = 3000
AGE_PEN    = 0.004
PERT_STR   = 0.30     # forza perturbazione polo→φ

# ── GESTALT ───────────────────────────────────────────────────────
FUSE_R      = MIN_D * 1.8   # raggio fusione — locale, non globale
FUSE_PH_THR = math.pi / 2.2 # soglia coerenza di fase
FUSE_MIN    = 3              # poli minimi per formare gestalt
FUSE_COH    = 0.65           # coerenza minima per stabilità gestalt
MERGE_R     = 3              # fusione solo da quasi sovrapposti
MERGE_PH    = math.pi / 3.5 # soglia fase per fusione gestalt
# SuperGestalt
SG_MIN_D    = 3.0   # distanza minima per SG — no SG tra gestalt quasi sovrapposti
SG_R        = 16    # zona intermedia
SG_PH_THR   = math.pi / 3.0  # soglia coerenza di fase per accoppiamento
SG_MIN_FRAMES = 30  # frame di coerenza sostenuta
SG_COUPLING = 0.008 # forza accoppiamento di fase top-down
SG_MERGE_PREVENT_R = 4  # gestalt in stesso SG non si fondono se distanza > questa
GEST_PERT   = 0.45           # forza perturbazione gestalt→φ
GEST_STEP   = 0.8            # velocità navigazione gestalt

# ── GENOMA ────────────────────────────────────────────────────────
G_LEARN    = 0.0008
G_MUT      = 0.04
G_POP_EMA  = 0.003

# ── PARAMETRO D'ORDINE ────────────────────────────────────────────
BETA_O     = 0.025

# ── CAMERA ────────────────────────────────────────────────────────
CAM_INIT_SCALE = 7.0   # pixel per unità griglia

# ═════════════════════════════════════════════════════════════════
# STRUTTURE DATI
# ═════════════════════════════════════════════════════════════════

_next_id = 0
def new_id():
    global _next_id; _next_id += 1; return _next_id

def random_genome():
    return np.array([
        np.random.uniform(0, 2*math.pi),  # g_ph
        np.random.uniform(0.2, 0.6),      # g_mu
        np.random.uniform(0.6, 1.4),      # g_noise
        1.0,                               # g_T_sens
        np.random.uniform(0, 2*math.pi),  # g_morph0
        np.random.uniform(0, 2*math.pi),  # g_morph1
        np.random.uniform(0, 2*math.pi),  # g_morph2
    ])

def mutate_genome(g):
    m = g.copy()
    m[0] = (m[0] + np.random.uniform(-G_MUT, G_MUT) * 2*math.pi) % (2*math.pi)
    m[1] = np.clip(m[1] + np.random.uniform(-G_MUT, G_MUT), 0.05, 0.95)
    m[2] = np.clip(m[2] + np.random.uniform(-G_MUT, G_MUT), 0.3, 1.8)
    m[3] = np.clip(m[3] + np.random.uniform(-G_MUT, G_MUT), 0.5, 3.0)
    for k in range(4, 7):
        m[k] = (m[k] + np.random.uniform(-G_MUT, G_MUT) * 2*math.pi) % (2*math.pi)
    return m


@dataclass
class Pole:
    id: int
    gx: float
    gy: float
    ph: float
    genome: np.ndarray
    vitality: float = 0.65
    age: int = 0
    T: float = 0.0
    T_ema: float = 0.0
    tau: float = 0.0
    mu: float = 0.3
    kap: float = 0.15
    fitness_ema: float = 0.5
    stuck: int = 0
    parent_id: int = -1
    lineage: int = -1
    morph_ph: np.ndarray = field(default_factory=lambda: np.random.uniform(0,2*math.pi,3))
    r_vis: float = 3.0
    dead: bool = False
    # Stato fusione: il polo converge verso il centroide prima di dissolversi
    merging: bool = False          # in fase di fusione
    merge_target_x: float = 0.0   # centroide destinazione
    merge_target_y: float = 0.0
    merge_frames: int = 0          # frame rimanenti prima di dissolversi


@dataclass
class Gestalt:
    """Organismo nato dalla fusione di poli."""
    id: int
    gx: float           # posizione in unità griglia
    gy: float
    ph: float           # fase collettiva
    genome: np.ndarray  # genotipo collettivo
    morph_ph: np.ndarray# firma morfologica (cristallizzata alla fusione)
    n_poles: int        # numero di poli fusi
    vitality: float = 0.7
    T: float = 0.0
    dissolving: int = 0    # se >0: sta rilasciando poli gradualmente
    poles_to_release: int = 0  # poli ancora da rilasciare
    r_vis: float = 10.0
    r_target: float = 10.0
    vx: float = 0.0
    vy: float = 0.0
    dead: bool = False
    lineage: int = -1
    birth_frame: int = 0   # frame di nascita — per calcolare durata vita
    T_life_avg: float = 0.0  # media EMA di T_locale durante la vita
    T_life_n: int = 0        # contatore frame per media
    T_pre_sg: float = 0.0    # T_avg accumulato PRIMA di entrare in SG
    T_pre_sg_n: int = 0      # frame accumulati prima di SG
    T_post_sg: float = 0.0   # T_avg accumulato DOPO ingresso in SG
    T_post_sg_n: int = 0     # frame accumulati dopo SG
    sg_entry_frame: int = -1 # frame di ingresso nel primo SG
    # SuperGestalt: accoppiamento relazionale senza fusione
    sg_id: int = -1          # id del supergestalt di appartenenza (-1 = libero)
    sg_coh_frames: int = 0   # frame di coerenza sostenuta con un partner
    sg_cooldown: int = 0     # frame di attesa dopo dissoluzione SG



@dataclass
class SuperGestalt:
    """Relazione emergente tra gestalt coerenti.
    Non è un oggetto aggiunto — è una struttura relazionale che diventa visibile.
    I gestalt membri mantengono identità distinta ma acquisiscono accoppiamento di fase."""
    id: int
    member_ids: list      # ids dei gestalt membri
    ph: float             # fase collettiva del supergestalt
    coh: float = 0.0      # coerenza attuale tra i membri
    age: int = 0          # frame di vita
    dead: bool = False

# ═════════════════════════════════════════════════════════════════
# CAMPO
# ═════════════════════════════════════════════════════════════════

class Field:
    def __init__(self):
        self.Ph  = np.random.uniform(0, 2*math.pi, (G, G)).astype(np.float32)
        self.Psi = np.random.uniform(-0.3, 0.3, (G, G)).astype(np.float32)
        self.Mu  = np.zeros((G, G), np.float32)
        self.Kap = np.ones((G, G), np.float32) * 0.15
        self.Vort= np.zeros((G, G), np.float32)
        self.T   = np.zeros((G, G), np.float32)
        self.Noise = np.ones((G, G), np.float32) * NOISE_BASE
        self.T_global = 0.0
        self.T_avg    = 0.0

    def step(self, noise_gain: float):
        # ── 1. φ: Kuramoto + Langevin
        Ph = self.Ph
        roll_r = np.roll(Ph, -1, axis=1)
        roll_l = np.roll(Ph,  1, axis=1)
        roll_u = np.roll(Ph, -1, axis=0)
        roll_d = np.roll(Ph,  1, axis=0)
        coup = KUR_K * (np.sin(roll_r-Ph) + np.sin(roll_l-Ph) +
                        np.sin(roll_u-Ph) + np.sin(roll_d-Ph)) * 0.25
        eta = np.random.standard_normal((G, G)).astype(np.float32) * 0.816
        self.Ph = (Ph + DT*coup + noise_gain*self.Noise*SQDT*eta) % (2*math.pi)

        # ── 2. Vorticity (plaquette 2×2)
        # CRITICO: ogni differenza di fase va wrappata INDIVIDUALMENTE a [-π,π]
        # prima di sommare. Il wrap sulla somma annulla il segnale topologico.
        Ph = self.Ph
        PI2f = float(2*math.pi)
        def wa(d):
            return d - PI2f * np.round(d / PI2f)
        Ph_r  = np.roll(Ph, -1, axis=1)   # x+1
        Ph_ur = np.roll(Ph_r, -1, axis=0)  # x+1, y+1
        Ph_u  = np.roll(Ph,  -1, axis=0)   # y+1
        Q  = wa(Ph_r  - Ph)
        Q += wa(Ph_ur - Ph_r)
        Q += wa(Ph_u  - Ph_ur)
        Q += wa(Ph    - Ph_u)
        self.Vort = (Q / PI2f).astype(np.float32)

        # ── 3. R_φ: ordine locale
        sc = np.cos(Ph)
        ss = np.sin(Ph)
        sc5 = sc + np.roll(sc,-1,axis=1)+np.roll(sc,1,axis=1)+np.roll(sc,-1,axis=0)+np.roll(sc,1,axis=0)
        ss5 = ss + np.roll(ss,-1,axis=1)+np.roll(ss,1,axis=1)+np.roll(ss,-1,axis=0)+np.roll(ss,1,axis=0)
        R_phi = np.sqrt(sc5**2 + ss5**2) / 5.0

        # ── 4. ψ: Allen-Cahn
        Psi = self.Psi
        lap = (np.roll(Psi,-1,axis=1)+np.roll(Psi,1,axis=1)+
               np.roll(Psi,-1,axis=0)+np.roll(Psi,1,axis=0) - 4*Psi)
        dPsi = DT*(PSI_D*lap - PSI_KAPPA*Psi + PSI_ALPHA*(R_phi - 0.5))
        self.Psi = np.clip(Psi + dPsi, -1, 1)

        # ── 5. Mu_ψ, Kap_ψ
        self.Mu = self.Mu*(1-0.03) + R_phi*0.03
        np.clip(self.Mu, 0, 1, out=self.Mu)
        psi_abs = np.abs(self.Psi)
        self.Kap = np.where(psi_abs > 0.70,
                            np.minimum(0.9, self.Kap + 0.15*DT),
                            np.maximum(0.05, self.Kap - 0.10*DT))

        # ── 6. T (latente)
        self.T = np.abs(self.Vort) * (1 - self.Kap) * (0.5 + self.Mu)
        self.T_global = float(self.T.mean())
        self.T_avg    = self.T_avg * 0.98 + self.T_global * 0.02

    def T_at(self, gx: float, gy: float) -> float:
        """T massimo in finestra 3×3 attorno al punto."""
        x = int(round(gx)); y = int(round(gy))
        xs = slice(max(0,x-1), min(G,x+2))
        ys = slice(max(0,y-1), min(G,y+2))
        return float(self.T[ys, xs].max()) if self.T[ys,xs].size else 0.0

    def T_in_direction(self, gx, gy, angle, step):
        """T massimo in finestra 3×3 in una direzione."""
        tx = int(round(gx + math.cos(angle)*step))
        ty = int(round(gy + math.sin(angle)*step))
        tx = max(0, min(G-1, tx)); ty = max(0, min(G-1, ty))
        xs = slice(max(0,tx-1), min(G,tx+2))
        ys = slice(max(0,ty-1), min(G,ty+2))
        return float(self.T[ys, xs].max()) if self.T[ys,xs].size else 0.0

    def perturb_phi(self, gx, gy, target_ph, strength, radius=1):
        """Polo/Gestalt pertuba φ introducendo disordine strutturato.
        Non spinge verso una fase target (ordinamento) ma crea gradienti
        di fase attorno al polo — condizione che genera vortici locali."""
        x = int(round(gx)); y = int(round(gy))
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                xi = max(0, min(G-1, x+dx)); yi = max(0, min(G-1, y+dy))
                if dx == 0 and dy == 0:
                    # Centro: piccola rotazione della fase corrente
                    self.Ph[yi, xi] = (self.Ph[yi, xi] + strength*0.4) % (2*math.pi)
                else:
                    # Vicini: perturbazione angolare proporzionale alla direzione
                    # → crea un gradiente radiale che favorisce la vorticity
                    angle = math.atan2(dy, dx)
                    self.Ph[yi, xi] = (self.Ph[yi, xi] +
                        strength * 0.25 * math.sin(angle - target_ph + self.Ph[yi,xi])*DT) % (2*math.pi)

    def deposit_psi(self, gx, gy, amount):
        """Deposita su ψ (decomposizione, dissoluzione)."""
        x = int(round(gx)); y = int(round(gy))
        if 0 <= x < G and 0 <= y < G:
            self.Psi[y,x] = float(np.clip(self.Psi[y,x] + amount, -1, 1))

    def shock(self):
        self.Ph = np.random.uniform(0, 2*math.pi, (G,G)).astype(np.float32)

    def seed(self, gx, gy, ph, radius=4):
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                x = int(round(gx+dx)); y = int(round(gy+dy))
                if 0 <= x < G and 0 <= y < G:
                    d = math.sqrt(dx**2+dy**2)
                    strength = max(0, 1 - d/radius)
                    self.Ph[y,x] = (self.Ph[y,x]*(1-strength*0.7) + ph*strength*0.7) % (2*math.pi)


# ═════════════════════════════════════════════════════════════════
# SIMULAZIONE
# ═════════════════════════════════════════════════════════════════

class Simulation:
    def __init__(self):
        self.field    = Field()
        self.poles: List[Pole] = []
        self.gestalts: List[Gestalt] = []
        self.supergestalts: List[SuperGestalt] = []
        self.pop_genome = random_genome()
        self.pop_gen    = 0
        self.o_t        = 0.0
        self.tau_global = 0.0
        self.tau_ema    = 0.0
        self.noise_gain = 1.0
        self.snap_count = 0
        self._snap_T    = 0.0
        self._frame     = 0
        self._spawn_initial()

    def _spawn_initial(self):
        cols = math.ceil(math.sqrt(N_START))
        rows = math.ceil(N_START / cols)
        gx_s = G/(cols+1); gy_s = G/(rows+1)
        n = 0
        for r in range(rows):
            for c in range(cols):
                if n >= N_START: break
                gx = gx_s*(c+1) + np.random.uniform(-gx_s*.3, gx_s*.3)
                gy = gy_s*(r+1) + np.random.uniform(-gy_s*.3, gy_s*.3)
                gx = max(2, min(G-2, gx)); gy = max(2, min(G-2, gy))
                g = random_genome()
                p = Pole(id=new_id(), gx=gx, gy=gy,
                         ph=np.random.uniform(0, 2*math.pi),
                         genome=g, morph_ph=np.random.uniform(0,2*math.pi,3))
                p.lineage = p.id
                self.poles.append(p); n += 1

    # ── STEP PRINCIPALE ──────────────────────────────────────────
    def step(self):
        self._frame += 1
        # Log T_global ogni 100 frame
        if self._frame % 100 == 0:
            n_poles = len(self.poles)
            n_gest  = len(self.gestalts)
            n_sg    = len(self.supergestalts)
            T_glob  = self.field.T_global
            noise   = self.noise_gain
            T_gest_local = sum(g.T for g in self.gestalts)/len(self.gestalts) if self.gestalts else 0
            print(f"[FIELD] frame={self._frame} T={T_glob:.5f} T_gest={T_gest_local:.5f} "
                  f"noise={noise:.3f} poles={n_poles} gest={n_gest} sg={n_sg}")
        f = self.field

        # 1. Aggiorna campo
        f.step(self.noise_gain)

        # 2. Parametro d'ordine non-Markoviano
        eps = (np.random.random() + np.random.random() - 1) * 0.006
        self.o_t = (1-BETA_O)*self.o_t + BETA_O*f.T_global + eps
        self.o_t = max(0, self.o_t)
        self.tau_global = self.o_t - f.T_global
        self.tau_ema    = self.tau_ema*0.97 + abs(self.tau_global)*0.03

        # 3. Control loop
        coh = self._coherence_poles()
        # Target: T naturale del campo senza agenti è 0.05-0.20
        crit_t = max(0.04, min(0.15, 0.06 + 0.08*(1-coh)))
        # T_sig: usa T_global direttamente (o_t lo insegue con ritardo)
        # T_std era troppo piccolo e falsava il segnale
        T_sig  = self.o_t
        err = crit_t - T_sig
        # Floor abbassato a 0.30 — il loop deve poter scendere
        # Guadagno 0.12 — risposta più rapida al crollo di T
        self.noise_gain = max(0.30, min(2.00, self.noise_gain + err * 0.12))
        f.Noise = f.Noise*0.999 + NOISE_BASE*0.001

        # 4. Entropy backpressure
        if coh > 0.87 and len(self.poles) < N_SOFT:
            excess = (coh - 0.87) / 0.13
            n_p = max(1, int(len(self.poles)*excess*0.4))
            sorted_p = sorted(self.poles, key=lambda p: p.vitality)
            for p in sorted_p[:n_p]:
                p.ph = (p.ph + np.random.uniform(-math.pi, math.pi)*excess*2) % (2*math.pi)
                ci_x = int(round(max(0,min(G-1,p.gx))))
                ci_y = int(round(max(0,min(G-1,p.gy))))
                f.Ph[ci_y,ci_x] = (f.Ph[ci_y,ci_x] + np.random.uniform(-math.pi,math.pi)*excess) % (2*math.pi)

        # 5. ColSnap
        if self._frame % 25 == 0:
            snap_thr = 0.010 * (1 - min(0.6, abs(self.tau_global)*10))
            if abs(f.T_global - self._snap_T) > snap_thr:
                self.snap_count += 1
                for p in self.poles: p.kap *= 0.75
                for g in self.gestalts: g.vitality *= 0.85
            self._snap_T = f.T_global

        # 6. Step poli
        self._step_poles()

        # 7. Step gestalt
        self._step_gestalts()

        # 8. Fusione poli → gestalt (ogni 8 frame — BFS è O(n²))
        if self._frame % 8 == 0:
            self._fuse_poles()

        # 9. Fusione gestalt → gestalt (ogni 15 frame)
        if self._frame % 15 == 0:
            self._merge_gestalts()

        # 10. Rilevamento SuperGestalt (ogni 15 frame)
        if self._frame % 15 == 0:
            self._detect_supergestalts()
        self._step_supergestalts()

        # 10. Perturbazioni su φ
        for p in self.poles:
            f.perturb_phi(p.gx, p.gy, p.genome[0], PERT_STR*p.vitality, radius=1)
        for g in self.gestalts:
            f.perturb_phi(g.gx, g.gy, g.genome[0], GEST_PERT*g.vitality, radius=2)

        # 11. Aggiorna pop_genome
        self._update_pop_genome()

    # ── STEP POLI ─────────────────────────────────────────────────
    def _step_poles(self):
        f = self.field
        pop_ratio   = len(self.poles) / N_SOFT
        pop_pressure= max(0, pop_ratio - 1.0)
        res_bonus   = max(0, (1-pop_ratio)*0.003)

        for p in self.poles:
            if p.dead: continue

            # T locale (finestra 3×3)
            p.T = f.T_at(p.gx, p.gy)
            p.T_ema = p.T_ema*(1-0.04) + p.T*0.04
            p.tau   = p.T - p.T_ema

            # Leggi campo
            ci_x = int(round(max(0,min(G-1,p.gx))))
            ci_y = int(round(max(0,min(G-1,p.gy))))
            p.mu  = float(f.Mu[ci_y,ci_x])
            p.kap = float(f.Kap[ci_y,ci_x])
            p.ph  = (p.ph*(1-0.08) + f.Ph[ci_y,ci_x]*0.08) % (2*math.pi)

            # Vitalità
            p.age += 1
            af    = min(1.0, p.age/AGE_MAX)
            decay = VIT_DECAY + AGE_PEN*af*af + pop_pressure*0.003
            p.vitality = max(0, min(1, p.vitality - decay + VIT_BASE + res_bonus + VIT_GAIN*p.T))

            if p.vitality < VIT_DEATH or p.age > AGE_MAX:
                f.deposit_psi(p.gx, p.gy, p.vitality*0.3)
                p.dead = True; continue

            # Apprendimento genomico
            p.fitness_ema = p.fitness_ema*0.995 + p.vitality*0.005
            if p.vitality > 0.55:
                lr = G_LEARN * p.vitality
                p.genome[0] = (p.genome[0]*(1-lr) + p.ph*lr) % (2*math.pi)
                p.genome[1] = np.clip(p.genome[1]*(1-lr)+p.mu*lr, 0.05, 0.95)
                p.genome[2] = np.clip(p.genome[2]*(1-lr)+float(f.Noise[ci_y,ci_x])*lr, 0.3, 1.8)

            # Riproduzione
            if not p.merging and p.vitality > VIT_REPRO and np.random.random() < REPRO_P and np.random.random() > pop_pressure:
                scarcity = max(0, 1 - len(self.poles)/N_SOFT)
                # Cerca direzione T massima
                best_a = np.random.uniform(0, 2*math.pi); best_T = 0.0
                for dd in range(8):
                    a = dd * math.pi/4
                    Td = f.T_in_direction(p.gx, p.gy, a, MIN_D*1.5)
                    if Td > best_T: best_T = Td; best_a = a
                spawn_d = MIN_D*1.5 + np.random.uniform(0, 3)
                cgx = max(2, min(G-2, p.gx + math.cos(best_a)*spawn_d))
                cgy = max(2, min(G-2, p.gy + math.sin(best_a)*spawn_d))
                child = Pole(
                    id=new_id(), gx=cgx, gy=cgy,
                    ph=(p.ph + np.random.uniform(-0.5,0.5)) % (2*math.pi),
                    genome=mutate_genome(p.genome),
                    morph_ph=np.random.uniform(0,2*math.pi,3),
                    vitality=0.55,
                )
                child.lineage = p.lineage
                child.parent_id = p.id
                self.poles.append(child)
                self.pop_gen += 1
                p.vitality *= 0.68

            # Navigazione (non se in fusione)
            if not p.merging:
                self._navigate_pole(p)

        # Rimuovi morti
        self.poles = [p for p in self.poles if not p.dead]

        # Respawn genomico
        while len(self.poles) < N_MIN:
            gx = np.random.uniform(2, G-2)
            gy = np.random.uniform(2, G-2)
            g  = mutate_genome(self.pop_genome)
            p  = Pole(id=new_id(), gx=gx, gy=gy,
                      ph=(self.pop_genome[0]+np.random.uniform(-0.6,0.6))%(2*math.pi),
                      genome=g, morph_ph=np.random.uniform(0,2*math.pi,3))
            p.lineage = p.id
            self.pop_gen += 1
            self.poles.append(p)

        # Separazione
        self._separate_poles()

    def _navigate_pole(self, p: Pole):
        f = self.field
        # 8 direzioni nel campo reale
        best_dx = 0.0; best_dy = 0.0; best_T = p.T
        for dd in range(8):
            a = dd * math.pi/4
            Td = f.T_in_direction(p.gx, p.gy, a, MIN_D*0.9)
            sv = Td * p.genome[3]
            if sv > best_T:
                best_T = sv; best_dx = math.cos(a); best_dy = math.sin(a)

        if best_dx or best_dy:
            gs = max(0.25, min(1.0, (best_T - p.T)*8 + 0.15))
            p.gx += best_dx * MAX_STEP * gs
            p.gy += best_dy * MAX_STEP * gs
            p.stuck = max(0, p.stuck-2)
        else:
            p.gx += np.random.uniform(-0.5, 0.5)
            p.gy += np.random.uniform(-0.5, 0.5)
            p.stuck += 1
            if p.stuck > 60:
                a = np.random.uniform(0, 2*math.pi)
                p.gx += math.cos(a) * MIN_D * 0.8
                p.gy += math.sin(a) * MIN_D * 0.8
                p.stuck = 0

        p.gx = max(1, min(G-1, p.gx))
        p.gy = max(1, min(G-1, p.gy))

    def _separate_poles(self):
        for _ in range(2):
            for i, pi in enumerate(self.poles):
                for pj in self.poles[i+1:]:
                    dx = pj.gx - pi.gx; dy = pj.gy - pi.gy
                    d2 = dx*dx + dy*dy
                    md = MIN_D * 0.55
                    if d2 < md*md and d2 > 0.0001:
                        d = math.sqrt(d2); push = (md-d)*0.5/d
                        pi.gx -= dx*push; pi.gy -= dy*push
                        pj.gx += dx*push; pj.gy += dy*push
            for p in self.poles:
                p.gx = max(1, min(G-1, p.gx))
                p.gy = max(1, min(G-1, p.gy))

    # ── FUSIONE POLI → GESTALT ────────────────────────────────────
    MERGE_ANIM_FRAMES = 18  # frame di animazione convergenza

    def _fuse_poles(self):
        """BFS su poli liberi: se ≥ FUSE_MIN poli coerenti → nasce un Gestalt."""
        free = [p for p in self.poles]
        if len(free) < FUSE_MIN: return

        visited = set()
        to_remove = set()
        R2 = FUSE_R * FUSE_R

        for i, seed in enumerate(free):
            if seed.id in visited: continue
            # BFS
            cluster = [seed]; queue = [seed]; visited.add(seed.id)
            qi = 0
            while qi < len(queue):
                cur = queue[qi]; qi += 1
                for other in free:
                    if other.id in visited: continue
                    dx = cur.gx-other.gx; dy = cur.gy-other.gy
                    if dx*dx+dy*dy < R2 and abs(_wa(cur.ph-other.ph)) < FUSE_PH_THR:
                        visited.add(other.id); cluster.append(other); queue.append(other)

            # Cap: max 8 poli per gestalt — cluster più grandi si dividono naturalmente
            cluster = cluster[:8]
            if len(cluster) < FUSE_MIN: continue

            # Coerenza interna — filtro rapido (stadio 1)
            sc = sum(math.cos(p.ph) for p in cluster)
            ss = sum(math.sin(p.ph) for p in cluster)
            coh = math.sqrt(sc*sc+ss*ss)/len(cluster)
            if coh < FUSE_COH: continue

            # Confine non degenere ∂(G) — conferma strutturale (stadio 2, ROF7)
            # Eseguito solo sui candidati che passano il filtro rapido.
            # ∂(G) = |<e^{iφ}>_interno - <e^{iφ}>_esterno| sul campo φ reale.
            # Il Gestalt esiste come referente distinto iff ∂(G) > θ_boundary.
            member_cells = set()
            for p in cluster:
                gx_i = int(round(max(0,min(G-1,p.gx))))
                gy_i = int(round(max(0,min(G-1,p.gy))))
                member_cells.add((gx_i,gy_i))
            sc_in=ss_in=0
            for (xi,yi) in member_cells:
                sc_in+=math.cos(self.field.Ph[yi,xi])
                ss_in+=math.sin(self.field.Ph[yi,xi])
            n_in=len(member_cells)
            v_in=complex(sc_in/n_in, ss_in/n_in) if n_in>0 else 0
            border_cells=set()
            for (xi,yi) in member_cells:
                for ddx,ddy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nb=(max(0,min(G-1,xi+ddx)),max(0,min(G-1,yi+ddy)))
                    if nb not in member_cells: border_cells.add(nb)
            if border_cells:
                sc_out=ss_out=0
                for (xi,yi) in border_cells:
                    sc_out+=math.cos(self.field.Ph[yi,xi])
                    ss_out+=math.sin(self.field.Ph[yi,xi])
                n_out=len(border_cells)
                v_out=complex(sc_out/n_out,ss_out/n_out)
                dG=abs(v_in-v_out)
            else:
                dG=0.0
            # θ_boundary=0.15: confine misurabile ma non troppo restrittivo
            if dG < 0.15:
                print(f"[REJECT_dG] frame={self._frame} coh={coh:.3f} dG={dG:.3f} "
                      f"n_poles={len(cluster)} -- passa COH, scartato da boundary")
                continue

            # Nasce il Gestalt
            cx = sum(p.gx for p in cluster)/len(cluster)
            cy = sum(p.gy for p in cluster)/len(cluster)
            ph_c = math.atan2(ss, sc) % (2*math.pi)
            genome_c = sum(p.genome*p.fitness_ema for p in cluster)
            genome_c /= sum(p.fitness_ema for p in cluster) + 1e-8
            genome_c = mutate_genome(genome_c)
            morph_c  = np.zeros(3)
            for pole in cluster:
                morph_c += pole.morph_ph / len(cluster)
            morph_c = morph_c % (2*math.pi)
            r_c = max(sum(math.hypot(p.gx-cx,p.gy-cy) for p in cluster)/len(cluster)+2, 4)

            g = Gestalt(
                id=new_id(), gx=cx, gy=cy, ph=ph_c,
                genome=genome_c, morph_ph=morph_c,
                n_poles=len(cluster),
                vitality=sum(p.vitality for p in cluster)/len(cluster),
                r_vis=min(14, r_c), r_target=min(14, r_c),
                lineage=cluster[0].lineage,
                birth_frame=self._frame,
            )
            # Log per verifica empirica post-hoc
            print(f"[FUSE] frame={self._frame} coh={coh:.3f} dG={dG:.3f} "
                  f"n_poles={len(cluster)} gestalt_id={g.id}")
            self.field.deposit_psi(cx, cy, 0.1)
            self.gestalts.append(g)
            self.pop_gen += 1

            for p in cluster:
                p.merging = True
                p.merge_target_x = cx
                p.merge_target_y = cy
                p.merge_frames = 18

        # I poli in merging convergono — vengono rimossi quando merge_frames==0
        still_alive = []
        for p in self.poles:
            if p.merging:
                p.merge_frames -= 1
                speed = 0.35 + (18 - p.merge_frames) * 0.04
                p.gx += (p.merge_target_x - p.gx) * speed
                p.gy += (p.merge_target_y - p.gy) * speed
                p.r_vis *= 0.88  # si rimpicciolisce
                if p.merge_frames > 0:
                    still_alive.append(p)
                # quando merge_frames==0: scompare (non aggiunto a still_alive)
            else:
                still_alive.append(p)
        self.poles = still_alive

    # ── STEP GESTALT ──────────────────────────────────────────────
    def _step_gestalts(self):
        f = self.field
        for g in self.gestalts:
            if g.dead: continue

            # T locale
            g.T = f.T_at(g.gx, g.gy)
            # Media cumulativa di T durante la vita
            g.T_life_n += 1
            g.T_life_avg += (g.T - g.T_life_avg) / g.T_life_n
            # T prima/dopo ingresso SG — test causalità
            if g.sg_id < 0:
                # Non ancora in SG: accumula T_pre
                g.T_pre_sg_n += 1
                g.T_pre_sg += (g.T - g.T_pre_sg) / g.T_pre_sg_n
            else:
                # In SG: registra frame ingresso e accumula T_post
                if g.sg_entry_frame < 0:
                    g.sg_entry_frame = self._frame
                g.T_post_sg_n += 1
                g.T_post_sg += (g.T - g.T_post_sg) / g.T_post_sg_n

            # Bilancio vitalità continuo — calibrato su T_locale_gestalt osservato
            # T_gest medio ~0.72, T_gest min ~0.51
            # T* deve stare tra 0.51 e 0.72 per produrre mortalità nelle fasi povere
            # T* = decay/(VIT_GAIN×gain_f) → gain_f = decay/(VIT_GAIN×T*)
            # Con T* = 0.60: gain_f = 0.0014/(0.12×0.60) = 0.0194
            gain  = VIT_GAIN * 0.0194 * g.T
            decay = VIT_DECAY * 1.0
            dV    = gain - decay
            g.vitality = max(0.0, min(1.0, g.vitality + dV))

            # Morte emergente: vitalità a zero
            if g.vitality <= 0.0:
                lifetime = self._frame - g.birth_frame
                sg_str = f"sg={g.sg_id}" if g.sg_id >= 0 else "sg=none"
                t_pre  = f"{g.T_pre_sg:.4f}" if g.T_pre_sg_n > 0 else "na"
                t_post = f"{g.T_post_sg:.4f}" if g.T_post_sg_n > 0 else "na"
                print(f"[DISS] frame={self._frame} causa=vitalita_zero vita={lifetime}f "
                      f"T={g.T:.4f} T_avg={g.T_life_avg:.4f} "
                      f"T_pre={t_pre} T_post={t_post} {sg_str}")
                self._dissolve(g)  # avvia dissoluzione graduale

            # Dissoluzione graduale: un polo per frame
            if g.dissolving > 0:
                self._release_one_pole(g)
                continue  # non naviga mentre si dissolve

            # Fase: piccolo drift
            ci_x = int(round(max(0,min(G-1,g.gx))))
            ci_y = int(round(max(0,min(G-1,g.gy))))
            g.ph = (g.ph*(1-0.05) + f.Ph[ci_y,ci_x]*0.05) % (2*math.pi)

            # Navigazione: 8 direzioni nel campo reale
            best_a = np.random.uniform(0,2*math.pi); best_T = g.T
            for dd in range(8):
                a = dd*math.pi/4
                Td = f.T_in_direction(g.gx, g.gy, a, g.r_vis*1.2)
                if Td*g.genome[3] > best_T:
                    best_T = Td*g.genome[3]; best_a = a
            gs = min(1.0, (best_T-g.T)*6+0.06)
            g.vx = g.vx*0.85 + math.cos(best_a)*GEST_STEP*gs*0.15
            g.vy = g.vy*0.85 + math.sin(best_a)*GEST_STEP*gs*0.15
            spd = math.hypot(g.vx, g.vy)
            if spd > GEST_STEP: g.vx*=GEST_STEP/spd; g.vy*=GEST_STEP/spd
            g.gx = max(2, min(G-2, g.gx+g.vx))
            g.gy = max(2, min(G-2, g.gy+g.vy))

            # Raggio visivo
            g.r_target = max(4, min(18, g.n_poles*1.2 + g.vitality*2))
            g.r_vis    = g.r_vis*0.92 + g.r_target*0.08

            # Morfologia evolve lentamente
            for k in range(3):
                g.morph_ph[k] = (g.morph_ph[k] + 0.0008*(k+1)) % (2*math.pi)

        self.gestalts = [g for g in self.gestalts if not g.dead]

    def _dissolve(self, g: Gestalt):
        """Avvia dissoluzione graduale: un polo per frame."""
        if g.dissolving == 0:
            g.poles_to_release = max(FUSE_MIN, g.n_poles)
            g.dissolving = g.poles_to_release

    def _release_one_pole(self, g: Gestalt):
        """Rilascia un singolo polo durante la dissoluzione graduale."""
        k = g.poles_to_release - g.dissolving
        n = g.poles_to_release
        a = k/n * 2*math.pi + np.random.uniform(-0.4, 0.4)
        d = g.r_vis * 0.5 + np.random.uniform(0, 3)
        gx = max(2, min(G-2, g.gx + math.cos(a)*d))
        gy = max(2, min(G-2, g.gy + math.sin(a)*d))
        p = Pole(id=new_id(), gx=gx, gy=gy,
                 ph=(g.ph+np.random.uniform(-0.5,0.5))%(2*math.pi),
                 genome=mutate_genome(g.genome),
                 morph_ph=np.random.uniform(0,2*math.pi,3),
                 vitality=max(0.25, g.vitality*0.65))
        p.lineage = g.lineage
        self.poles.append(p)
        g.dissolving -= 1
        if g.dissolving <= 0:
            self.field.deposit_psi(g.gx, g.gy, g.vitality*0.25)
            g.dead = True

    # ── FUSIONE GESTALT → GESTALT ─────────────────────────────────
    def _merge_gestalts(self):
        """Due gestalt vicini e compatibili si fondono in un super-organismo."""
        merged = set()
        new_g  = []
        for i, ga in enumerate(self.gestalts):
            if ga.id in merged: continue
            for gb in self.gestalts[i+1:]:
                if gb.id in merged: continue
                d = math.hypot(ga.gx-gb.gx, ga.gy-gb.gy)
                if d > MERGE_R: continue
                if abs(_wa(ga.ph-gb.ph)) > MERGE_PH: continue
                # Non fondere gestalt già in stesso SuperGestalt
                if ga.sg_id >= 0 and ga.sg_id == gb.sg_id: continue
                # Fusione
                w_a = ga.vitality*ga.n_poles; w_b = gb.vitality*gb.n_poles
                wt  = w_a + w_b + 1e-8
                cx  = (ga.gx*w_a + gb.gx*w_b)/wt
                cy  = (ga.gy*w_a + gb.gy*w_b)/wt
                sc  = math.cos(ga.ph)*w_a + math.cos(gb.ph)*w_b
                ss  = math.sin(ga.ph)*w_a + math.sin(gb.ph)*w_b
                ph_c= math.atan2(ss,sc) % (2*math.pi)
                genome_c = (ga.genome*w_a + gb.genome*w_b)/wt
                genome_c = mutate_genome(genome_c)
                morph_c  = (ga.morph_ph*w_a + gb.morph_ph*w_b)/wt % (2*math.pi)
                r_c = max(ga.r_vis, gb.r_vis)*1.15
                gm  = Gestalt(
                    id=new_id(), gx=cx, gy=cy, ph=ph_c,
                    genome=genome_c, morph_ph=morph_c,
                    n_poles=ga.n_poles+gb.n_poles,
                    vitality=(ga.vitality+gb.vitality)*0.5,
                    r_vis=min(14, r_c), r_target=min(14, r_c),
                    lineage=ga.lineage,
                )
                new_g.append(gm)
                merged.add(ga.id); merged.add(gb.id)
                self.pop_gen += 1
                break  # un gestalt può fondersi una volta per step
        self.gestalts = [g for g in self.gestalts if g.id not in merged] + new_g


    # ── SUPERGESTALT ──────────────────────────────────────────────
    def _detect_supergestalts(self):
        """Rileva coppie/gruppi di gestalt con coerenza di fase sostenuta.
        Non fusione — accoppiamento relazionale che persiste nel tempo."""
        if len(self.gestalts) < 2: return

        # Aggiorna contatori di coerenza tra coppie
        # Decrementa cooldown
        for g in self.gestalts:
            if g.sg_cooldown > 0: g.sg_cooldown -= 1

        for i, ga in enumerate(self.gestalts):
            if ga.dead or ga.sg_cooldown > 0: continue
            for gb in self.gestalts[i+1:]:
                if gb.dead or gb.sg_cooldown > 0: continue
                d = math.hypot(ga.gx-gb.gx, ga.gy-gb.gy)
                if d > SG_R or d < SG_MIN_D: continue
                dph = abs(_wa(ga.ph-gb.ph))
                if dph < SG_PH_THR:
                    # Coerenza sostenuta: incrementa contatore su entrambi
                    ga.sg_coh_frames = min(ga.sg_coh_frames+3, SG_MIN_FRAMES*2)
                    gb.sg_coh_frames = min(gb.sg_coh_frames+3, SG_MIN_FRAMES*2)
                else:
                    ga.sg_coh_frames = max(0, ga.sg_coh_frames-2)
                    gb.sg_coh_frames = max(0, gb.sg_coh_frames-2)

        # Forma SuperGestalt quando coerenza sufficiente
        existing_sg_members = {g.id for sg in self.supergestalts
                                for gid in sg.member_ids for g in self.gestalts
                                if g.id == gid}

        for i, ga in enumerate(self.gestalts):
            if ga.dead or ga.sg_coh_frames < SG_MIN_FRAMES: continue
            for gb in self.gestalts[i+1:]:
                if gb.dead or gb.sg_coh_frames < SG_MIN_FRAMES: continue
                if ga.sg_id >= 0 and ga.sg_id == gb.sg_id: continue  # già accoppiati
                d = math.hypot(ga.gx-gb.gx, ga.gy-gb.gy)
                if d > SG_R or d < SG_MIN_D: continue
                dph = abs(_wa(ga.ph-gb.ph))
                if dph > SG_PH_THR: continue

                # Forma nuovo SuperGestalt
                sc = math.cos(ga.ph)+math.cos(gb.ph)
                ss = math.sin(ga.ph)+math.sin(gb.ph)
                ph_sg = math.atan2(ss,sc) % (2*math.pi)
                coh = math.sqrt(sc*sc+ss*ss)/2
                sg = SuperGestalt(
                    id=new_id(),
                    member_ids=[ga.id, gb.id],
                    ph=ph_sg, coh=coh
                )
                self.supergestalts.append(sg)
                ga.sg_id = sg.id; gb.sg_id = sg.id
                print(f"[SG] Formato SuperGestalt {sg.id} — coh={coh:.2f} dist={math.hypot(ga.gx-gb.gx,ga.gy-gb.gy):.1f}")

    def _step_supergestalts(self):
        """Aggiorna SuperGestalt: fase collettiva e accoppiamento top-down."""
        alive_sg = []
        for sg in self.supergestalts:
            if sg.dead: continue
            sg.age += 1

            # Trova membri vivi
            members = [g for g in self.gestalts
                       if g.id in sg.member_ids and not g.dead]
            if len(members) < 2:
                # SuperGestalt si dissolve
                for g in self.gestalts:
                    if g.id in sg.member_ids:
                        g.sg_id = -1
                        g.sg_coh_frames = 0
                sg.dead = True; continue

            # Aggiorna fase collettiva
            sc = sum(math.cos(g.ph) for g in members)
            ss = sum(math.sin(g.ph) for g in members)
            sg.ph = math.atan2(ss,sc) % (2*math.pi)
            sg.coh = math.sqrt(sc*sc+ss*ss)/len(members)

            # Dissolvi se coerenza crolla
            if sg.coh < 0.3:
                for g in members:
                    g.sg_id = -1; g.sg_coh_frames = 0
                    g.sg_cooldown = 60  # ~2 secondi prima di riformare
                sg.dead = True; continue

            # Accoppiamento top-down: i membri derivano verso la fase collettiva
            for g in members:
                g.ph = (g.ph + SG_COUPLING*math.sin(sg.ph - g.ph)) % (2*math.pi)

            alive_sg.append(sg)

        # Cap: max 8 SuperGestalt attivi — tieni i più coerenti
        # (alzato da 3 per robustezza statistica dell'effetto SG sulla sopravvivenza)
        if len(alive_sg) > 8:
            alive_sg.sort(key=lambda sg: sg.coh, reverse=True)
            # Libera i gestalt esclusi
            kept_ids = {sg.id for sg in alive_sg[:8]}
            for sg in alive_sg[8:]:
                for g in self.gestalts:
                    if g.id in sg.member_ids and g.sg_id == sg.id:
                        g.sg_id = -1
                        g.sg_coh_frames = 0
            alive_sg = alive_sg[:8]
        self.supergestalts = alive_sg

    # ── POP GENOME ────────────────────────────────────────────────
    def _update_pop_genome(self):
        all_agents = self.poles + [_GestaltProxy(g) for g in self.gestalts]
        if not all_agents: return
        fw = 0; pg = np.zeros(7)
        for a in all_agents:
            w = a.fitness_ema; fw += w
            pg += a.genome * w
        if fw > 0:
            pg /= fw
            self.pop_genome = self.pop_genome*(1-G_POP_EMA) + pg*G_POP_EMA

    def _coherence_poles(self):
        if not self.poles: return 0.0
        sc = sum(math.cos(p.ph) for p in self.poles)
        ss = sum(math.sin(p.ph) for p in self.poles)
        return math.sqrt(sc*sc+ss*ss)/len(self.poles)

    @property
    def pop(self): return len(self.poles) + len(self.gestalts)


# Proxy per gestalt nel pop_genome update
class _GestaltProxy:
    def __init__(self, g): self.genome=g.genome; self.fitness_ema=g.vitality


def _wa(a):
    pi2 = 2*math.pi
    return a - pi2*round(a/pi2)


# ═════════════════════════════════════════════════════════════════
# RENDERING
# ═════════════════════════════════════════════════════════════════

def ph_to_rgb(ph, sat=0.7, lum=0.5):
    """Fase → colore RGB (0-255)."""
    h = ph/(2*math.pi)
    # HSL semplificato
    C = (1 - abs(2*lum-1))*sat
    X = C*(1 - abs((h*6)%2 - 1))
    m = lum - C/2
    hh = int(h*6)
    if   hh==0: r,g,b=C,X,0
    elif hh==1: r,g,b=X,C,0
    elif hh==2: r,g,b=0,C,X
    elif hh==3: r,g,b=0,X,C
    elif hh==4: r,g,b=X,0,C
    else:        r,g,b=C,0,X
    return (int((r+m)*255), int((g+m)*255), int((b+m)*255))


def organic_points(cx, cy, r, amp, ph, morph_ph, t, n=22):
    """Punti del profilo organico in coordinate pixel."""
    pts = []
    for k in range(n):
        theta = k/n * 2*math.pi
        rot = theta - ph*0.5
        dr  = sum(math.sin((j+1)*rot + morph_ph[j] + t*(j*0.35+0.25)) for j in range(3))
        dr *= amp/3
        rad = max(r*0.3, r+dr)
        pts.append((cx + rad*math.cos(theta), cy + rad*math.sin(theta)))
    return pts


class Camera:
    def __init__(self):
        self.ox = WIN_W/2 - G/2*CAM_INIT_SCALE
        self.oy = WIN_H/2 - G/2*CAM_INIT_SCALE
        self.scale = CAM_INIT_SCALE
        self._drag = None

    def world_to_screen(self, gx, gy):
        return (int(self.ox + gx*self.scale), int(self.oy + gy*self.scale))

    def screen_to_world(self, sx, sy):
        return ((sx-self.ox)/self.scale, (sy-self.oy)/self.scale)

    def zoom(self, factor, cx, cy):
        gx,gy = self.screen_to_world(cx,cy)
        self.scale = max(2.0, min(40.0, self.scale*factor))
        self.ox = cx - gx*self.scale
        self.oy = cy - gy*self.scale

    def pan(self, dx, dy):
        self.ox += dx; self.oy += dy

    def event(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button==1:
            self._drag = ev.pos
        elif ev.type == pygame.MOUSEBUTTONUP and ev.button==1:
            self._drag = None
        elif ev.type == pygame.MOUSEMOTION and self._drag:
            dx = ev.pos[0]-self._drag[0]; dy = ev.pos[1]-self._drag[1]
            self.pan(dx,dy); self._drag = ev.pos
        elif ev.type == pygame.MOUSEWHEEL:
            mx,my = pygame.mouse.get_pos()
            self.zoom(1.1 if ev.y>0 else 0.91, mx,my)


def draw_pole(surf, cam, p: Pole, t: float, scale: float):
    sx,sy = cam.world_to_screen(p.gx, p.gy)
    r = max(3.0, p.r_vis * scale/CAM_INIT_SCALE)
    col_main = ph_to_rgb(p.ph, 0.55+p.mu*0.3, 0.28+p.mu*0.28)
    col_core = ph_to_rgb(p.ph, 0.8,  0.65+p.mu*0.2)

    # Corpo con bordo luminoso — no Surface allocation
    pygame.draw.circle(surf, col_main, (sx,sy), int(r))
    # Alone: cerchio più grande con colore attenuato
    rim = (min(255,col_core[0]//2), min(255,col_core[1]//2), min(255,col_core[2]//2))
    pygame.draw.circle(surf, rim, (sx,sy), int(r*1.5), 1)

    # Alone τ pulsante
    if abs(p.tau) > 0.003:
        phase = (frame_counter * abs(p.tau) * 15) % 1.0
        rr = int(r + phase*r*1.5)
        ra = int((1-phase)*80*p.vitality*min(1,abs(p.tau)*20))
        if ra > 5:
            pygame.draw.circle(surf, (*col_core, ra), (sx,sy), rr, 1)

    # Nucleo
    pygame.draw.circle(surf, col_core, (sx,sy), max(2, int(r*0.35)))


def draw_gestalt(surf, cam, g: Gestalt, t: float, scale: float):
    sx,sy = cam.world_to_screen(g.gx, g.gy)
    r = max(6.0, g.r_vis * scale/CAM_INIT_SCALE)
    col_main = ph_to_rgb(g.ph, 0.5,  0.22+g.vitality*0.2)
    col_core = ph_to_rgb(g.ph, 0.85, 0.65)
    col_rim  = ph_to_rgb(g.ph, 0.7,  0.50)
    # Tinta SG: se membro di SuperGestalt, deriva verso bianco-argento
    if g.sg_id >= 0:
        blend = 0.50
        col_main = tuple(min(255, int(v*(1-blend) + 210*blend)) for v in col_main)
        col_rim  = (200, 220, 255)
        col_core = tuple(min(255, int(v*(1-blend) + 255*blend)) for v in col_core)

    # Anelli concentrici — no Surface allocation
    for ring_r, ring_col in [
        (int(r*1.8), (col_main[0]//4, col_main[1]//4, col_main[2]//4)),
        (int(r*1.4), (col_main[0]//2, col_main[1]//2, col_main[2]//2)),
    ]:
        pygame.draw.circle(surf, ring_col, (sx,sy), ring_r, 1)

    # Anello esterno (confine organismo)
    pygame.draw.circle(surf, col_rim, (sx,sy), int(r), max(1,int(g.n_poles*0.4)))

    # Corpo
    pygame.draw.circle(surf, col_main, (sx,sy), int(r*0.82))

    # Nucleo pulsante
    pulse = 0.85 + 0.15*math.sin(frame_counter*0.08 + g.ph)
    pygame.draw.circle(surf, col_core, (sx,sy), max(3, int(r*0.28*pulse)))

    # Label n_poles (font passato come arg)
    if r > 12:
        lbl = _font_sm.render(str(g.n_poles), True, (180,230,200))
        surf.blit(lbl, (sx+int(r*0.45), sy-8))



def draw_supergestalt(surf, cam, sg, gestalts_map, t, font_sm):
    """SuperGestalt come relazione visibile.
    Connessioni bezier animate tra gestalt membri.
    Anelli sincronizzati sui gestalt membri."""
    members = [g for g in gestalts_map if g.id in sg.member_ids and not g.dead]
    if len(members) < 2: return

    col_sg = ph_to_rgb(sg.ph, 0.3, 0.75)  # bianco-argento, bassa saturazione
    alpha_sg = int(120 * sg.coh)

    # Connessioni bezier tra ogni coppia di membri
    for i, ga in enumerate(members):
        for gb in members[i+1:]:
            sxa,sya = cam.world_to_screen(ga.gx, ga.gy)
            sxb,syb = cam.world_to_screen(gb.gx, gb.gy)
            # Punto di controllo oscillante — riflette la dinamica del campo
            mid_x = (sxa+sxb)//2
            mid_y = (sya+syb)//2
            # Oscillazione perpendicolare alla connessione
            dx = sxb-sxa; dy = syb-sya
            length = math.hypot(dx,dy)+0.001
            perp_x = -dy/length; perp_y = dx/length
            osc = math.sin(t*0.8 + sg.ph) * length * 0.25 * (1-sg.coh)
            cx = int(mid_x + perp_x*osc)
            cy = int(mid_y + perp_y*osc)
            # Approssimazione bezier quadratica con segmenti
            pts = []
            N = 12
            for k in range(N+1):
                u = k/N
                # Formula quadratica: (1-u)²P0 + 2u(1-u)Pc + u²P1
                bx = int((1-u)**2*sxa + 2*u*(1-u)*cx + u**2*sxb)
                by = int((1-u)**2*sya + 2*u*(1-u)*cy + u**2*syb)
                pts.append((bx,by))
            if len(pts) >= 2:
                lw = max(2, int(3*sg.coh))  # più spesse
                # Colore più saturo — bianco-argento luminoso
                bright = min(255, int(180 + 75*sg.coh))
                c_bright = (bright, bright, min(255, bright+30))
                pygame.draw.lines(surf, c_bright, False, pts, lw)
                # Secondo passaggio: linea più sottile e più luminosa al centro
                if lw >= 3:
                    c_core = (255, 255, 255)
                    pygame.draw.lines(surf, c_core, False, pts, 1)

    # Anello sincronizzato sui gestalt membri — doppio anello pulsante
    pulse = 0.7 + 0.3*math.sin(t*1.2 + sg.ph)
    bright = min(255, int(180 + 75*sg.coh))
    ring_col_outer = (bright, bright, min(255,bright+30))
    ring_col_inner = (255, 255, 255)
    for g in members:
        sx,sy = cam.world_to_screen(g.gx, g.gy)
        base_r = int(g.r_vis * cam.scale/CAM_INIT_SCALE)
        r_outer = int(base_r * 1.7 * pulse)
        r_inner = int(base_r * 1.4 * pulse)
        if r_outer > 3:
            pygame.draw.circle(surf, ring_col_outer, (sx,sy), r_outer, 2)
        if r_inner > 2:
            pygame.draw.circle(surf, ring_col_inner, (sx,sy), r_inner, 1)

# ═════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════

frame_counter = 0
_font_sm = None

def main():
    pygame.init()
    pygame.display.set_caption('SYMPHŌNŌN Ω — GESTALT v3')
    # Usa la risoluzione dello schermo reale
    info   = pygame.display.Info()
    sw, sh = info.current_w, info.current_h
    global WIN_W, WIN_H
    WIN_W, WIN_H = sw, sh
    surf   = pygame.display.set_mode((WIN_W, WIN_H), pygame.FULLSCREEN)
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont('monospace', 12)
    font_s = pygame.font.SysFont('monospace', 10)
    global _font_sm; _font_sm = pygame.font.SysFont('monospace', 10)

    sim    = Simulation()
    cam    = Camera()
    # Trail: sfondo persistente scuro
    trail  = pygame.Surface((WIN_W, WIN_H))
    trail.fill((2,2,10))
    # Fade layer pre-allocato — riusato ogni frame
    fade_layer = pygame.Surface((WIN_W, WIN_H))
    fade_layer.fill((2,2,10))
    fade_layer.set_alpha(65)

    paused  = False
    frame_n = 0
    fps_acc = 0.0

    running = True
    while running:
        dt_ms = clock.tick(FPS_TARGET)
        fps_acc = fps_acc*0.95 + (1000/(dt_ms+0.001))*0.05

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT: running=False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE: running=False
                elif ev.key == pygame.K_SPACE: paused = not paused
                elif ev.key == pygame.K_r:
                    sim = Simulation(); trail.fill((2,2,10)); frame_n=0
                elif ev.key == pygame.K_s:
                    sim.field.shock()
                elif ev.key == pygame.K_d:
                    sim.field.seed(G/2, G/2, np.random.uniform(0,2*math.pi))
            cam.event(ev)

        if not paused:
            sim.step()
            frame_n += 1
            global frame_counter; frame_counter = frame_n

        # ── Rendering ──────────────────────────────────────────
        # Trail: dissolvenza lenta
        trail.blit(fade_layer,(0,0))  # riusa surface pre-allocata

        t = frame_n * 0.005

        # Connessioni tra poli vicini
        drawn = set()
        for i,pi in enumerate(sim.poles):
            for pj in sim.poles[i+1:]:
                key = (min(pi.id,pj.id), max(pi.id,pj.id))
                if key in drawn: continue
                d = math.hypot(pi.gx-pj.gx, pi.gy-pj.gy)
                if d > FUSE_R*1.2: continue
                dphi = abs(_wa(pi.ph-pj.ph))
                if dphi > math.pi*0.55: continue
                coh  = 1 - dphi/(math.pi*0.55)
                alpha= int(coh*60*min(pi.mu,pj.mu)*min(pi.vitality,pj.vitality))
                if alpha < 5: continue
                col = ph_to_rgb((pi.ph+pj.ph)/2, 0.5, 0.55)
                lw  = max(1, int(coh*2))
                sx1,sy1 = cam.world_to_screen(pi.gx,pi.gy)
                sx2,sy2 = cam.world_to_screen(pj.gx,pj.gy)
                # Draw diretto — no Surface allocation per linea
                r_a = min(255, col[0]+alpha//2)
                g_a = min(255, col[1]+alpha//2)
                b_a = min(255, col[2]+alpha//2)
                pygame.draw.line(trail, (r_a,g_a,b_a), (sx1,sy1),(sx2,sy2),lw)
                drawn.add(key)

        # Poli
        for p in sim.poles:
            draw_pole(trail, cam, p, t, cam.scale)

        # Gestalt
        for g in sim.gestalts:
            draw_gestalt(trail, cam, g, t, cam.scale)

        # ── SuperGestalt (strato relazionale — sopra i gestalt)
        for sg in sim.supergestalts:
            draw_supergestalt(trail, cam, sg, sim.gestalts, t, _font_sm)

        surf.blit(trail,(0,0))

        # ── HUD ──────────────────────────────────────────────
        T_disp = f'{sim.field.T_global:.5f}'
        lines = [
            ('SYMPHŌNŌN Ω · GESTALT v4', (100,160,130)),
            (f'POP  {len(sim.poles)}p · {len(sim.gestalts)}g · {len(sim.supergestalts)}sg  Φ{sim.pop_gen}', (160,210,180)),
            (f'T̄   {T_disp}', (140,190,160)),
            (f'τ   {sim.tau_ema:.4f}', (130,180,150)),
            (f'GAIN {sim.noise_gain:.2f}  FLIP {sim.snap_count}', (120,170,140)),
            (f'FPS  {int(fps_acc)}', (80,120,100)),
        ]
        if paused:
            lines.append(('— PAUSA —', (200,150,80)))
        y = 10
        for txt,col in lines:
            s = font.render(txt, True, col)
            surf.blit(s,(12,y)); y+=16

        # Legenda tasti
        hints = ['SPACE=pausa  R=reset  S=shock  D=seed  ESC=esci',
                 'drag=pan  scroll=zoom']
        for i,h in enumerate(hints):
            s = font_s.render(h, True, (60,90,75))
            surf.blit(s,(12, WIN_H-22+i*12))

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    main()
