"""
SYMPHONON O v17 - COHERENCE PROBE + PARAMETRO D'ORDINE

Novita rispetto a v16:
  o(t+1) = o(t) + beta*(lambda2(t) - o(t)) + epsilon
  tension = o - lambda2  (inerzia / crollo imminente)
  o_stability = 1 - std(o)  (attrattore raggiunto se > 0.80)
  Test: std(o) < std(lambda2) --> memoria reale confermata
  Nuova riga dashboard + quarta riga nel plot matplotlib

Uso:
  python symphonon_probe_v17.py
  python symphonon_probe_v17.py --episodes 200 --agents 8
  python symphonon_probe_v17.py --probe
  python symphonon_probe_v17.py --no-plot

Deps: numpy  (matplotlib opzionale)
"""

import numpy as np
import sys
import time
import argparse
from collections import deque, defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

try:
    import torch
    TORCH = True
except ImportError:
    TORCH = False


# ===============================================================
# ANSI
# ===============================================================
class A:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    WHITE   = "\033[97m"
    ORANGE  = "\033[38;5;214m"

def ansi(text, *codes):
    return "".join(codes) + text + A.RESET

def bar(value, width=20, fill="X", empty="."):
    # use ascii-safe chars, caller can override
    n = int(np.clip(value, 0.0, 1.0) * width)
    return fill * n + empty * (width - n)

def bar_u(value, width=20):
    n = int(np.clip(value, 0.0, 1.0) * width)
    return "\u2588" * n + "\u2591" * (width - n)

def sparkline(values, width=40, lo=0.0, hi=1.0):
    chars = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
    if len(values) == 0:
        return "\u2500" * width
    tail = list(values)[-width:]
    out = []
    for v in tail:
        idx = int(np.clip((v - lo) / (hi - lo + 1e-9), 0, 1) * (len(chars) - 1))
        out.append(chars[idx])
    return "".join(out)


# ===============================================================
# CONFIG  (identico a v16)
# ===============================================================
class Config:
    N_AGENTS         = 12
    TAU_DIM          = 16
    OBS_DIM          = 20
    ACTION_DIM       = 5
    HIDDEN_DIM       = 64

    F_ALPHA          = 0.10
    F_BETA           = 0.04
    F_GAMMA          = 0.08
    F_DELTA          = 0.06
    F_LAMBDA         = 0.25

    LANGEVIN_DT      = 0.015
    D_BASE           = 0.010
    D_PSI_SCALE      = 0.80

    TAU_CLIP         = 1.20
    TAU_FLOOR        = 0.015

    SOC_SCORE_THR    = 0.02
    EMA_LONG         = 0.050
    EMA_SHORT        = 0.150
    SCORE_LONG_W     = 0.65

    REGIME_COS_THR   = 0.35
    REGIME_MIN_SIZE  = 2

    ANCHOR_RRI_THR   = 0.40
    ANCHOR_PULL_THR  = 0.30
    ANCHOR_BETA      = 0.04

    LOW_RRI_THR      = 0.05
    LOW_RRI_K        = 80
    RECRYSTAL_DUR    = 120
    RECRYSTAL_FACT   = 0.50

    LOCK_CURV_MID    = 0.12
    LOCK_CURV_SCALE  = 0.03
    LOCK_MIN         = 0.30
    LOCK_ALI_BOOST   = 0.70

    ANCHOR_DECAY     = 0.9980
    ANCHOR_MAX       = 8
    ANCHOR_COMPETE_W = 0.025
    ANCHOR_REPEL_W   = 0.008

    LR_POLICY        = 3e-3
    TAU_POLICY_ALPHA = 5e-4
    GAMMA_RL         = 0.97
    ENTROPY_COEF     = 0.01
    TAU_ALIGN_W      = 0.30

    N_EPISODES       = 1200
    EPISODE_LEN      = 60
    LOG_EVERY        = 40
    LYAP_WINDOW      = 50

    GRID_SIZE        = 10
    FOOD_REWARD      = 1.0
    MOVE_COST        = 0.01

C = Config()


# ===============================================================
# COHERENCE JUDGE  (v17: + parametro d'ordine)
# ===============================================================
class CoherenceJudge:
    """
    Giudice di coerenza relazionale v17.

    lambda2 = W_RRI*rri + W_PLU*(1-psi) + W_GFC*gfc + W_REW*reward_norm

    Parametro d'ordine (NOVITA' v17):
        o(t+1) = o(t) + O_BETA*(lambda2(t) - o(t)) + epsilon

    Proprieta':
      - lambda2 stabile alto  -> o converge lentamente (attrattore reale)
      - lambda2 oscilla       -> o filtra (memoria smorzante non-markoviana)
      - shock su lambda2      -> o NON segue subito (inerzia = traccia persistente)

    Test di falsificabilita':
        std(o) < std(lambda2)  -> memoria reale confermata
    """

    W_RRI  = 0.40
    W_PLU  = 0.25
    W_GFC  = 0.20
    W_REW  = 0.15

    THR_CRYSTAL  = 0.65
    THR_CRITICAL = 0.40
    TREND_WINDOW = 20

    O_BETA  = 0.15   # accoppiamento: piu' basso = piu' inerzia
    O_NOISE = 0.02   # rumore stocastico residuo

    def __init__(self):
        self._lambda2_hist = deque(maxlen=self.TREND_WINDOW)
        self._f_hist       = deque(maxlen=self.TREND_WINDOW)
        self._o            = 0.5
        self._o_hist       = deque(maxlen=self.TREND_WINDOW)

    def evaluate(self, psi, rri, gfc, reward_norm, f_val=None):
        pluralism = 1.0 - psi

        lambda2 = float(np.clip(
            self.W_RRI * rri + self.W_PLU * pluralism
            + self.W_GFC * gfc + self.W_REW * reward_norm,
            0.0, 1.0
        ))

        rri_j = rri * pluralism
        drift = max(0.0, 0.35 - lambda2)

        if lambda2 > self.THR_CRYSTAL:
            state = "CRISTALLIZZATO"
            color = A.CYAN
            icon  = "O="
            msg   = "Struttura relazionale stabile. Regimi distinti e persistenti."
        elif lambda2 > self.THR_CRITICAL:
            state = "CRITICO"
            color = A.YELLOW
            icon  = "O~"
            msg   = "Zona di biforcazione. Competing attractors - esito ancora aperto."
        else:
            state = "INSTABILE"
            color = A.RED
            icon  = "O!"
            msg   = "Campo in fusione. Deriva caotica o dissoluzione dei regimi."

        self._lambda2_hist.append(lambda2)
        if f_val is not None:
            self._f_hist.append(f_val)

        trend = 0.0
        if len(self._lambda2_hist) >= 4:
            xs = np.arange(len(self._lambda2_hist), dtype=float)
            ys = np.array(self._lambda2_hist, dtype=float)
            trend = float(np.polyfit(xs, ys, 1)[0])

        forecast = "-> stabile"
        if   trend >  0.003: forecast = "-> cristallizzazione"
        elif trend < -0.003: forecast = "-> fusione"

        # --- parametro d'ordine dinamico ---
        noise    = np.random.uniform(-self.O_NOISE, self.O_NOISE)
        self._o  = float(np.clip(
            self._o + self.O_BETA * (lambda2 - self._o) + noise,
            0.0, 1.0
        ))
        self._o_hist.append(self._o)

        # tension = o - lambda2
        #   > 0 : o conserva memoria alta nonostante lambda2 sia sceso (inerzia)
        #   < 0 : lambda2 e' salito, o deve ancora raggiungerlo
        tension = float(self._o - lambda2)

        # stabilita' di o: vicino a 1 = attrattore raggiunto
        o_stability = float(1.0 - np.std(list(self._o_hist))) if len(self._o_hist) > 2 else 0.5

        return dict(
            lambda2     = lambda2,
            rri_j       = rri_j,
            drift       = drift,
            state       = state,
            color       = color,
            icon        = icon,
            msg         = msg,
            trend       = trend,
            forecast    = forecast,
            pluralism   = pluralism,
            o           = self._o,
            tension     = tension,
            o_stability = o_stability,
            psi         = psi,
            rri         = rri,
            gfc         = gfc,
            reward_norm = reward_norm,
        )

    def reset(self):
        self._lambda2_hist.clear()
        self._f_hist.clear()
        self._o_hist.clear()
        self._o = 0.5


# ===============================================================
# POLICY  (identica a v16)
# ===============================================================
class LinearPolicy:
    def __init__(self):
        self.theta       = np.random.randn(C.OBS_DIM, C.ACTION_DIM).astype(np.float32) * 0.05
        self.tau_proj    = np.random.randn(C.TAU_DIM, C.ACTION_DIM).astype(np.float32) * 0.02
        self._last_obs   = None
        self._last_act   = -1
        self._last_probs = None

    def _forward(self, obs, tau):
        return obs @ self.theta + tau @ self.tau_proj

    def act(self, obs_np, tau_np):
        logits = self._forward(obs_np, tau_np)
        logits -= logits.max()
        probs   = np.exp(logits) / (np.exp(logits).sum() + 1e-8)
        action  = np.random.choice(C.ACTION_DIM, p=probs)
        log_p   = float(np.log(probs[action] + 1e-8))
        ent     = float(-np.sum(probs * np.log(probs + 1e-8)))
        self._last_obs   = obs_np.copy()
        self._last_act   = action
        self._last_probs = probs.copy()
        return action, log_p, ent, probs

    def update(self, reward, mean_reward, tau, psi):
        if self._last_obs is None:
            return
        obs   = self._last_obs
        a     = self._last_act
        probs = self._last_probs
        advantage  = float(reward) - float(mean_reward)
        grad_log   = -probs.copy()
        grad_log[a] += 1.0
        self.theta += C.LR_POLICY * advantage * np.outer(obs, grad_log)
        psi_gate   = 1.0 - 1.0 / (1.0 + np.exp(-(float(psi) - 0.30) / 0.08))
        alpha      = C.TAU_POLICY_ALPHA * psi_gate
        tau_signal = tau @ self.tau_proj
        tn = float(np.linalg.norm(tau_signal)) + 1e-8
        self.theta += alpha * np.outer(obs, tau_signal / tn)
        np.clip(self.theta, -2.0, 2.0, out=self.theta)

PolicyNet = LinearPolicy


# ===============================================================
# AGENT  (identico a v16)
# ===============================================================
class Agent:
    _uid_counter = 0

    def __init__(self):
        self.uid          = Agent._uid_counter
        Agent._uid_counter += 1
        self.policy       = PolicyNet()
        self.tau          = np.random.randn(C.TAU_DIM).astype(np.float32) * 0.3
        self._tau_enforce()
        self.social_mem   = {}
        self._ali_score   = 0.0
        self._ant_score   = 0.0
        self.tau_curv     = 0.02
        self.regime_idx   = -1
        self.ep_log_probs = []
        self.ep_rewards   = []
        self.ep_entropies = []
        self.ep_tau_aligns= []

    def _tau_enforce(self):
        n = float(np.linalg.norm(self.tau))
        if   n > C.TAU_CLIP:          self.tau *= C.TAU_CLIP / n
        elif 1e-9 < n < C.TAU_FLOOR:  self.tau *= C.TAU_FLOOR / n
        elif n <= 1e-9:
            self.tau = np.random.randn(C.TAU_DIM).astype(np.float32) * C.TAU_FLOOR

    def relation_score(self, uid):
        if uid not in self.social_mem:
            return 0.0
        l, s = self.social_mem[uid]
        return l * C.SCORE_LONG_W + s * (1.0 - C.SCORE_LONG_W)

    def update_social(self, other_uid, outcome):
        prev = self.social_mem.get(other_uid, [0.0, 0.0])
        ln   = prev[0] * (1 - C.EMA_LONG)  + outcome * C.EMA_LONG
        sn   = prev[1] * (1 - C.EMA_SHORT) + outcome * C.EMA_SHORT
        self.social_mem[other_uid] = [ln, sn]

    def reset_episode(self):
        self.ep_log_probs  = []
        self.ep_rewards    = []
        self.ep_entropies  = []
        self.ep_tau_aligns = []


# ===============================================================
# REGIME DETECTOR  (identico a v16)
# ===============================================================
class RegimeDetector:

    @staticmethod
    def detect(agents):
        n = len(agents)
        if n == 0:
            return np.array([], dtype=int), np.array([]), np.zeros((0, C.TAU_DIM)), 0, -1, 0.0
        taus    = np.array([a.tau for a in agents], dtype=np.float32)
        tau_hat = taus / (np.linalg.norm(taus, axis=1, keepdims=True) + 1e-8)
        labels  = np.full(n, -1, dtype=int)
        centroids = []
        cent_norm = []
        for i in range(n):
            if not centroids:
                labels[i] = 0
                centroids.append(tau_hat[i].copy())
                cent_norm.append(tau_hat[i].copy())
                continue
            sims = np.array([float(np.dot(tau_hat[i], c)) for c in cent_norm])
            best = int(np.argmax(sims))
            if sims[best] >= C.REGIME_COS_THR:
                labels[i] = best
                k = int(np.sum(labels == best))
                centroids[best] = (centroids[best] * (k - 1) + tau_hat[i]) / k
                cn = float(np.linalg.norm(centroids[best]))
                cent_norm[best] = centroids[best] / (cn + 1e-8)
            else:
                nid = len(centroids)
                labels[i] = nid
                centroids.append(tau_hat[i].copy())
                cent_norm.append(tau_hat[i].copy())
        nc    = len(centroids)
        sizes = np.array([int(np.sum(labels == k)) for k in range(nc)])
        sig   = np.where(sizes >= C.REGIME_MIN_SIZE)[0]
        nr    = len(sig)
        if nr == 0:
            return labels, sizes, np.array(cent_norm), 0, -1, 0.0
        di = int(sig[np.argmax(sizes[sig])])
        df = float(sizes[di]) / n
        return labels, sizes, np.array(cent_norm), nr, di, df

    @staticmethod
    def compute_psi(sizes, n_total):
        sig = sizes[sizes >= C.REGIME_MIN_SIZE]
        K   = len(sig)
        if K <= 1 or n_total == 0:
            return 1.0 if K <= 1 else 0.0
        p    = sig / sig.sum()
        H    = -np.sum(p * np.log(p + 1e-12))
        Hmax = np.log(K)
        return float(1.0 - H / Hmax) if Hmax > 0 else 1.0

    @staticmethod
    def compute_rri(agents, labels, dom_idx):
        if dom_idx < 0 or not agents:
            return 0.0
        members = [a for i, a in enumerate(agents)
                   if i < len(labels) and labels[i] == dom_idx]
        if not members:
            return 0.0
        return float(np.clip(
            1.0 - np.mean([m.tau_curv for m in members]) / (C.TAU_FLOOR * 4),
            0.0, 1.0
        ))

    @staticmethod
    def compute_gfc(taus):
        if len(taus) < 3:
            return 0.0
        tn = taus / (np.linalg.norm(taus, axis=1, keepdims=True) + 1e-8)
        return float(np.sqrt(np.mean(tn[:, 0])**2 + np.mean(tn[:, 1])**2))

    @staticmethod
    def compute_psi_p(agents):
        if len(agents) < 2:
            return 0.5
        taus = np.array([a.tau for a in agents], dtype=np.float32)
        tn   = taus / (np.linalg.norm(taus, axis=1, keepdims=True) + 1e-8)
        n    = min(len(agents), 8)
        idx  = np.random.choice(len(agents), n, replace=False)
        cv   = [float(np.dot(tn[idx[i]], tn[idx[j]]))
                for i in range(n) for j in range(i + 1, n)]
        return float((1.0 - np.mean(cv)) * 0.5) if cv else 0.5


# ===============================================================
# VARIATIONAL FIELD  (identico a v16)
# ===============================================================
class VariationalField:
    _anchors       = {}
    _low_rri_count = 0
    _recrystal_on  = False
    _recrystal_cnt = 0

    @staticmethod
    def _softmax(v):
        e = np.exp(v - v.max())
        return e / (e.sum() + 1e-8)

    @staticmethod
    def compute_F(psi, rri, gfc, psi_p, reward_norm=0.0):
        return (C.F_ALPHA * psi + C.F_BETA * rri
                - C.F_GAMMA * gfc - C.F_DELTA * psi_p
                - C.F_LAMBDA * reward_norm)

    @staticmethod
    def local_gradients(agent, agents, mu_pop):
        tau_i = agent.tau
        dPsi  = tau_i - mu_pop
        sw    = VariationalField._softmax(np.abs(tau_i))
        dRRI  = sw * np.sign(tau_i + 1e-9)
        lap   = np.zeros(C.TAU_DIM, dtype=np.float32)
        n_nb  = 0
        for other in agents:
            if other.uid == agent.uid:
                continue
            sc = agent.relation_score(other.uid)
            if abs(sc) > C.SOC_SCORE_THR:
                if sc > 0:
                    lap += (other.tau - tau_i) * sc
                else:
                    lap -= (other.tau - tau_i) * abs(sc) * 0.8
                n_nb += 1
        if n_nb > 0:
            lap /= n_nb
        dG    = -lap
        p     = VariationalField._softmax(np.abs(tau_i))
        dPsiL = -(np.log(p + 1e-9) + 1.0) * np.sign(tau_i + 1e-9)
        return (C.F_ALPHA * dPsi + C.F_BETA * dRRI
                + C.F_GAMMA * dG  + C.F_DELTA * dPsiL)

    @staticmethod
    def langevin_step(agents, psi, rri, gfc, psi_p, reward_norm=0.0):
        if not agents:
            return
        vf     = VariationalField
        mu_pop = np.mean([a.tau for a in agents], axis=0)

        for agent in agents:
            ali_a = ant_a = ali_c = ant_c = 0
            for other in agents:
                if other.uid == agent.uid:
                    continue
                sc = agent.relation_score(other.uid)
                if   sc > 0: ali_a += sc;  ali_c += 1
                elif sc < 0: ant_a += -sc; ant_c += 1
            agent._ali_score = ali_a / max(ali_c, 1)
            agent._ant_score = ant_a / max(ant_c, 1)

        regime_taus = {}
        for agent in agents:
            rid = agent.regime_idx
            if rid < 0:
                continue
            regime_taus.setdefault(rid, []).append(agent.tau)

        if rri > 0.25:
            ema_rate = float(np.clip((rri - 0.25) / 0.50, 0.05, 0.40))
            for rid, taus in regime_taus.items():
                mean_t = np.mean(taus, axis=0).astype(np.float32)
                if rid in vf._anchors:
                    vf._anchors[rid] = vf._anchors[rid] * (1 - ema_rate) + mean_t * ema_rate
                else:
                    vf._anchors[rid] = mean_t.copy()
            vf._low_rri_count = 0
        else:
            vf._low_rri_count += 1

        decay_eff = C.ANCHOR_DECAY + (1.0 - C.ANCHOR_DECAY) * (
            1.0 / (1.0 + np.exp(-(rri - 0.30) / 0.08)))
        for rid in list(vf._anchors.keys()):
            vf._anchors[rid] *= decay_eff
            if float(np.linalg.norm(vf._anchors[rid])) < C.TAU_FLOOR:
                del vf._anchors[rid]

        if len(vf._anchors) > C.ANCHOR_MAX:
            norms = {rid: float(np.linalg.norm(a)) for rid, a in vf._anchors.items()}
            keep  = sorted(norms, key=norms.get, reverse=True)[:C.ANCHOR_MAX]
            vf._anchors = {rid: vf._anchors[rid] for rid in keep}

        recryst_strength = 1.0 / (1.0 + np.exp(-(0.15 - rri) / 0.05))
        r_factor = 1.0 - 0.60 * recryst_strength
        if rri < C.LOW_RRI_THR:
            vf._low_rri_count += 1
            vf._recrystal_on = (vf._low_rri_count > 5)
        else:
            vf._low_rri_count = max(0, vf._low_rri_count - 1)
            vf._recrystal_on = False

        D_eff         = C.D_BASE * (0.25 + float(psi) * C.D_PSI_SCALE) * r_factor
        cycle_inertia = 1.0 / (1.0 + np.exp(-(rri - 0.40) / 0.08))
        D_eff        *= (1.0 - 0.35 * cycle_inertia)
        noise_std     = float(np.sqrt(2.0 * D_eff * C.LANGEVIN_DT))
        nucl_gate     = 1.0 / (1.0 + np.exp((rri - 0.15) / 0.04))
        pull_boost    = 1.0 + 7.0 * nucl_gate

        for agent in agents:
            tau_old = agent.tau.copy()
            grad_F  = VariationalField.local_gradients(agent, agents, mu_pop)

            last_r  = float(agent.ep_rewards[-1]) if agent.ep_rewards else 0.0
            surplus = float(np.clip(last_r - reward_norm, -0.5, 0.5))
            if abs(surplus) > 0.005:
                tau_dir = tau_old / (float(np.linalg.norm(tau_old)) + 1e-8)
                grad_F  = grad_F - tau_dir * surplus * C.F_LAMBDA * 0.5

            own_rid = agent.regime_idx
            for rid, anchor in vf._anchors.items():
                if rid == own_rid:
                    pull   = C.ANCHOR_COMPETE_W * pull_boost * (anchor - agent.tau)
                    grad_F = grad_F - pull
                else:
                    repel_gate = 1.0 / (1.0 + np.exp(-(rri - 0.20) / 0.06))
                    diff  = agent.tau - anchor
                    dn    = float(np.linalg.norm(diff)) + 1e-8
                    repel = C.ANCHOR_REPEL_W * repel_gate * (diff / dn)
                    grad_F = grad_F + repel

            curv_dist = (C.LOCK_CURV_MID - agent.tau_curv) / C.LOCK_CURV_SCALE
            sig  = 1.0 / (1.0 + np.exp(-curv_dist))
            lock = C.LOCK_MIN + (1.0 - C.LOCK_MIN) * (1.0 - sig)
            if agent._ali_score > agent._ant_score + 0.01:
                lock *= C.LOCK_ALI_BOOST

            noise_scale = lock * (1.0 - 0.65 * nucl_gate)
            noise  = np.random.randn(C.TAU_DIM).astype(np.float32) * noise_std * noise_scale
            agent.tau = tau_old - lock * C.LANGEVIN_DT * grad_F + noise
            agent.tau = 0.92 * agent.tau + 0.08 * tau_old
            agent._tau_enforce()

            to_n = float(np.linalg.norm(tau_old))
            tn_n = float(np.linalg.norm(agent.tau))
            if to_n > 1e-6 and tn_n > 1e-6:
                raw_c = float(np.linalg.norm(agent.tau / tn_n - tau_old / to_n))
            else:
                raw_c = 0.0
            agent.tau_curv = agent.tau_curv * 0.96 + raw_c * 0.04


# ===============================================================
# FORAGING GRID  (identico a v16)
# ===============================================================
class ForagingGrid:
    N_PATCHES     = 4
    PATCH_SIZE    = 2
    PATCH_REWARDS = [1.0, 0.8, 1.2, 0.9]
    RESPAWN_RATE  = 0.15

    def __init__(self):
        g = C.GRID_SIZE
        self.G = g
        self.patch_centers = np.array([
            [g // 4,     g // 4],
            [3 * g // 4, g // 4],
            [g // 4,     3 * g // 4],
            [3 * g // 4, 3 * g // 4],
        ])
        self.patch_cells     = []
        self.patch_cell_sets = []
        for cx, cy in self.patch_centers:
            cells = [((cx + dx) % g, (cy + dy) % g)
                     for dx in range(-self.PATCH_SIZE, self.PATCH_SIZE + 1)
                     for dy in range(-self.PATCH_SIZE, self.PATCH_SIZE + 1)]
            self.patch_cells.append(cells)
            self.patch_cell_sets.append(set(cells))
        self.reset()

    def _cell_patch(self, x, y):
        for p, pset in enumerate(self.patch_cell_sets):
            if (int(x), int(y)) in pset:
                return p
        return -1

    def reset(self):
        g = self.G
        self.pos = np.zeros((C.N_AGENTS, 2), dtype=int)
        for i in range(C.N_AGENTS):
            p = i % self.N_PATCHES
            cx, cy = self.patch_centers[p]
            self.pos[i] = [(cx + np.random.randint(-1, 2)) % g,
                           (cy + np.random.randint(-1, 2)) % g]
        self.food = np.zeros((g, g), dtype=np.float32)
        for p, cells in enumerate(self.patch_cells):
            rv = self.PATCH_REWARDS[p]
            for cx, cy in cells:
                if np.random.rand() < 0.7:
                    self.food[cx, cy] = rv
        self.reward_ema = np.zeros(C.N_AGENTS)
        self.prev_pos   = self.pos.copy()

    def _obs(self, i):
        x, y = self.pos[i]
        g    = self.G
        obs  = np.zeros(C.OBS_DIM, dtype=np.float32)
        obs[0] = x / g;  obs[1] = y / g
        obs[2] = float((x - self.prev_pos[i, 0]) / g)
        obs[3] = float((y - self.prev_pos[i, 1]) / g)
        dists = [float(np.linalg.norm(self.patch_centers[p] - np.array([x, y])) / g)
                 for p in range(self.N_PATCHES)]
        sp = np.argsort(dists)
        obs[4] = dists[sp[0]];  obs[5] = float(self.patch_centers[sp[0], 0] - x) / g
        obs[6] = dists[sp[1]];  obs[7] = float(self.patch_centers[sp[1], 0] - x) / g
        cp = self._cell_patch(x, y)
        if 0 <= cp < 4:
            obs[8 + cp] = 1.0
        n_near = sum(1 for j in range(C.N_AGENTS)
                     if j != i and np.linalg.norm(self.pos[j] - self.pos[i]) < 2.5)
        obs[12] = n_near / C.N_AGENTS
        obs[13] = float(self.reward_ema[i])
        obs[14] = float(cp) / self.N_PATCHES if cp >= 0 else -0.1
        obs[15] = float(n_near) / 5.0
        for ki, (ddx, ddy) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
            obs[16 + ki] = self.food[(x + ddx) % g, (y + ddy) % g]
        return obs

    def step(self, actions):
        self.prev_pos = self.pos.copy()
        g = self.G
        rewards = np.full(C.N_AGENTS, -C.MOVE_COST)
        dx_map = {0: -1, 1: 1, 2: 0, 3: 0, 4: 0}
        dy_map = {0:  0, 1: 0, 2: -1, 3: 1, 4: 0}
        for i, a in enumerate(actions):
            if a < 4:
                self.pos[i, 0] = (self.pos[i, 0] + dx_map[a]) % g
                self.pos[i, 1] = (self.pos[i, 1] + dy_map[a]) % g
            else:
                x, y = self.pos[i]
                if self.food[x, y] > 0:
                    cp   = self._cell_patch(x, y)
                    same = sum(1 for j in range(C.N_AGENTS)
                               if j != i and self._cell_patch(*self.pos[j]) == cp and cp >= 0)
                    rewards[i] += max(float(self.food[x, y]) - same * 0.08, 0.1)
                    self.food[x, y] = 0.0
        for p, cells in enumerate(self.patch_cells):
            rv = self.PATCH_REWARDS[p]
            for cx, cy in cells:
                if self.food[cx, cy] == 0.0 and np.random.rand() < self.RESPAWN_RATE:
                    self.food[cx, cy] = rv
        self.reward_ema = self.reward_ema * 0.95 + rewards * 0.05
        return rewards

    def get_all_obs(self):
        return [self._obs(i) for i in range(C.N_AGENTS)]


# ===============================================================
# TRAINING LOOP
# ===============================================================
def train(n_episodes=None, n_agents=None, seed=42):
    np.random.seed(seed)
    if n_episodes: C.N_EPISODES = n_episodes
    if n_agents:   C.N_AGENTS   = n_agents

    judge = CoherenceJudge()
    W     = 70

    print()
    print(ansi("=" * W, A.CYAN))
    print(ansi("  SYMPHONON O v17 -- COHERENCE PROBE + PARAMETRO D'ORDINE", A.BOLD, A.CYAN))
    print(ansi("  Agenti: {}  | tau-dim: {}  | Episodi: {}".format(
        C.N_AGENTS, C.TAU_DIM, C.N_EPISODES), A.DIM))
    print(ansi("  O_BETA={}  O_NOISE={}".format(
        CoherenceJudge.O_BETA, CoherenceJudge.O_NOISE), A.DIM))
    print(ansi("=" * W, A.CYAN))

    Agent._uid_counter = 0
    agents = [Agent() for _ in range(C.N_AGENTS)]
    env    = ForagingGrid()
    rd     = RegimeDetector()

    VariationalField._anchors       = {}
    VariationalField._low_rri_count = 0
    VariationalField._recrystal_on  = False
    VariationalField._recrystal_cnt = 0

    lyap_win = deque(maxlen=C.LYAP_WINDOW)
    f_prev   = 0.0

    history = dict(
        F=[], psi=[], n_regimes=[], gfc=[],
        reward=[], lyap=[], rri=[], df_dt=[],
        lambda2=[], state=[], rri_j=[], drift=[],
        o=[], tension=[], o_stability=[]
    )

    t0 = time.time()

    for episode in range(C.N_EPISODES):
        env.reset()
        for a in agents:
            a.reset_episode()
        ep_rewards_all = []

        rl, sizes, _, n_reg, dom_idx, _ = rd.detect(agents)
        psi   = rd.compute_psi(sizes, len(agents))
        rri   = rd.compute_rri(agents, rl, dom_idx)
        gfc   = rd.compute_gfc(np.array([a.tau for a in agents]))
        psi_p = rd.compute_psi_p(agents)

        for step in range(C.EPISODE_LEN):
            obs_list   = env.get_all_obs()
            tau_mean   = np.mean([a.tau for a in agents], axis=0)
            tau_mean_n = tau_mean / (np.linalg.norm(tau_mean) + 1e-8)
            actions, log_ps, ents = [], [], []
            for i, agent in enumerate(agents):
                act, lp, ent, _ = agent.policy.act(obs_list[i], agent.tau)
                actions.append(act); log_ps.append(lp); ents.append(ent)
            rewards = env.step(actions)
            ep_rewards_all.append(rewards.mean())
            for i, agent in enumerate(agents):
                agent.ep_log_probs.append(log_ps[i])
                agent.ep_rewards.append(float(rewards[i]))
                agent.ep_entropies.append(ents[i])
                tn_i = agent.tau / (np.linalg.norm(agent.tau) + 1e-8)
                agent.ep_tau_aligns.append(float(np.dot(tn_i, tau_mean_n)))
            for i, a in enumerate(agents):
                for j, b in enumerate(agents):
                    if i >= j: continue
                    pi = env._cell_patch(*env.pos[i])
                    pj = env._cell_patch(*env.pos[j])
                    compete   = 1.0 if (pi == pj and pi >= 0) else 0.1
                    outcome_a = rewards[i] - float(rewards[j]) * 0.6 * compete - 0.05 * compete
                    outcome_b = rewards[j] - float(rewards[i]) * 0.6 * compete - 0.05 * compete
                    a.update_social(b.uid, outcome_a)
                    b.update_social(a.uid, outcome_b)

        mean_r      = float(np.mean(ep_rewards_all))
        reward_norm = float(np.clip(mean_r / C.FOOD_REWARD, 0, 1))

        rl, sizes, _, n_reg, dom_idx, dom_frac = rd.detect(agents)
        psi   = rd.compute_psi(sizes, len(agents))
        rri   = rd.compute_rri(agents, rl, dom_idx)
        gfc   = rd.compute_gfc(np.array([a.tau for a in agents]))
        psi_p = rd.compute_psi_p(agents)

        for i, a in enumerate(agents):
            a.regime_idx = int(rl[i]) if i < len(rl) else -1

        f_val   = VariationalField.compute_F(psi, rri, gfc, psi_p, reward_norm)
        df_dt   = f_val - f_prev; f_prev = f_val
        lyap_win.append(1 if df_dt < 0 else 0)
        lyap_pct = float(np.mean(list(lyap_win))) if lyap_win else 0.5

        VariationalField.langevin_step(agents, psi, rri, gfc, psi_p, reward_norm)

        mean_r_ep = float(np.mean(ep_rewards_all))
        for agent in agents:
            agent_r = float(np.mean(agent.ep_rewards)) if agent.ep_rewards else mean_r_ep
            agent.policy.update(agent_r, mean_r_ep, agent.tau, psi)

        jud = judge.evaluate(psi, rri, gfc, reward_norm, f_val)

        history['F'].append(f_val)
        history['psi'].append(psi)
        history['n_regimes'].append(n_reg)
        history['gfc'].append(gfc)
        history['reward'].append(mean_r)
        history['lyap'].append(lyap_pct)
        history['rri'].append(rri)
        history['df_dt'].append(df_dt)
        history['lambda2'].append(jud['lambda2'])
        history['state'].append(jud['state'])
        history['rri_j'].append(jud['rri_j'])
        history['drift'].append(jud['drift'])
        history['o'].append(jud['o'])
        history['tension'].append(jud['tension'])
        history['o_stability'].append(jud['o_stability'])

        if (episode + 1) % C.LOG_EVERY == 0:
            elapsed   = time.time() - t0
            recryst   = " [RECRYST]" if VariationalField._recrystal_on else ""
            n_anch    = len(VariationalField._anchors)
            anch_s    = " [A:{}]".format(n_anch) if n_anch else ""
            l2_color  = (A.CYAN   if jud['state'] == "CRISTALLIZZATO"
                         else A.YELLOW if jud['state'] == "CRITICO" else A.RED)
            o_color   = (A.CYAN   if jud['o'] > 0.65
                         else A.YELLOW if jud['o'] > 0.40 else A.RED)
            ten_col   = (A.RED    if jud['tension'] < -0.05
                         else A.GREEN if jud['tension'] > 0.05 else A.DIM)
            psi_c     = A.RED if psi > 0.7 else A.YELLOW if psi > 0.5 else A.GREEN

            spark_l2 = sparkline(history['lambda2'], width=26)
            spark_o  = sparkline(history['o'],       width=26)

            print()
            hdr = "  ep {:>5d}/{}".format(episode + 1, C.N_EPISODES)
            print(ansi(hdr, A.BOLD, A.WHITE)
                  + ansi("  t={}s".format(int(elapsed)), A.DIM)
                  + ansi(recryst + anch_s, A.MAGENTA))

            # riga 1: campo
            print(
                "  F={}  dF={}{}  Psi={}  RRI={}  GFC={}  R={}  reg={}  LYAP={}".format(
                    ansi("{:>+7.4f}".format(f_val), A.BLUE),
                    ansi("{:>+7.4f}".format(df_dt), A.GREEN if df_dt < 0 else A.RED),
                    "v" if df_dt < 0 else "^",
                    ansi("{:.3f}".format(psi), psi_c),
                    ansi("{:.3f}".format(rri), A.CYAN),
                    ansi("{:.3f}".format(gfc), A.BLUE),
                    ansi("{:>+6.3f}".format(mean_r), A.GREEN if mean_r > 0 else A.RED),
                    ansi(str(n_reg), A.MAGENTA),
                    ansi("{:.0f}%".format(lyap_pct * 100),
                         A.GREEN if lyap_pct > 0.55 else A.YELLOW),
                )
            )

            # riga 2: lambda2
            l2_s    = "{:.4f}".format(jud['lambda2'])
            drift_s = "{:.3f}".format(jud['drift'])
            drift_c = A.YELLOW if jud['drift'] > 0.1 else A.DIM
            print(
                "  {} {:<16s}  l2={}  [{}]  drift={}  {}".format(
                    jud['icon'],
                    ansi(jud['state'], l2_color, A.BOLD),
                    ansi(l2_s, l2_color),
                    ansi(bar_u(jud['lambda2'], 14), l2_color),
                    ansi(drift_s, drift_c),
                    ansi(jud['forecast'], A.DIM),
                )
            )

            # riga 3: parametro d'ordine
            o_s     = "{:.4f}".format(jud['o'])
            ten_s   = "{:+.4f}".format(jud['tension'])
            ostab_s = "{:.3f}".format(jud['o_stability'])
            ostab_c = A.GREEN if jud['o_stability'] > 0.8 else A.DIM
            print(
                "  l2[{}]  o[{}]".format(
                    ansi(spark_l2, l2_color),
                    ansi(spark_o, o_color),
                )
            )
            print(
                "     o={}  [{}]  tension={}  o_stab={}".format(
                    ansi(o_s, o_color),
                    ansi(bar_u(jud['o'], 14), o_color),
                    ansi(ten_s, ten_col),
                    ansi(ostab_s, ostab_c),
                )
            )

    print()
    print(ansi("=" * W, A.CYAN))
    print(ansi("  TRAINING COMPLETO", A.BOLD, A.GREEN))
    print(ansi("=" * W, A.CYAN))

    return agents, history, judge


# ===============================================================
# SUMMARY
# ===============================================================
def summary(history, agents):
    n    = len(history['F'])
    last = max(1, n // 5)
    rd   = RegimeDetector()

    print()
    print(ansi("  RIEPILOGO -- ultimi {} episodi".format(last), A.BOLD, A.WHITE))
    print()

    l2_last   = float(np.mean(history['lambda2'][-last:]))
    rri_last  = float(np.mean(history['rri'][-last:]))
    gfc_last  = float(np.mean(history['gfc'][-last:]))
    psi_last  = float(np.mean(history['psi'][-last:]))
    lyap_g    = float(np.mean(history['lyap']))
    rew_last  = float(np.mean(history['reward'][-last:]))
    n_g       = float(np.mean(history['n_regimes'][-last:]))

    states  = history['state'][-last:]
    n_cryst = states.count("CRISTALLIZZATO")
    n_crit  = states.count("CRITICO")
    n_inst  = states.count("INSTABILE")

    print("  l2 medio finale:   {}   [{}]".format(
        ansi("{:.4f}".format(l2_last), A.CYAN), bar_u(l2_last, 20)))
    print("  RRI medio:         {}".format(ansi("{:.4f}".format(rri_last), A.CYAN)))
    print("  GFC medio:         {}".format(ansi("{:.4f}".format(gfc_last), A.BLUE)))
    print("  Psi medio:         {}".format(ansi("{:.4f}".format(psi_last), A.YELLOW)))
    print("  Reward medio:      {}".format(
        ansi("{:+.4f}".format(rew_last), A.GREEN if rew_last > 0 else A.RED)))
    print("  Lyapunov globale:  {}".format(
        ansi("{:.1f}%".format(lyap_g * 100), A.GREEN if lyap_g > 0.55 else A.YELLOW)))
    print("  Regimi (media):    {}".format(ansi("{:.1f}".format(n_g), A.MAGENTA)))

    # --- parametro d'ordine ---
    o_last     = float(np.mean(history['o'][-last:]))
    ten_last   = float(np.mean(history['tension'][-last:]))
    ostab_last = float(np.mean(history['o_stability'][-last:]))
    std_l2     = float(np.std(history['lambda2'][-last:]))
    std_o      = float(np.std(history['o'][-last:]))
    memory_ok  = std_o < std_l2

    print()
    print(ansi("  PARAMETRO D'ORDINE  o(t+1) = o + beta*(l2-o) + eps", A.BOLD, A.ORANGE))
    print("  o medio:           {}   [{}]".format(
        ansi("{:.4f}".format(o_last), A.ORANGE), bar_u(o_last, 20)))
    ten_c = A.GREEN if ten_last > 0.02 else A.RED if ten_last < -0.02 else A.DIM
    print("  tensione (o-l2):   {}".format(ansi("{:+.4f}".format(ten_last), ten_c)))
    print("  stabilita' di o:   {}".format(
        ansi("{:.4f}".format(ostab_last), A.GREEN if ostab_last > 0.8 else A.YELLOW)))
    mem_msg = ("   std(l2)={:.4f}  std(o)={:.4f}  {} memoria reale confermata".format(
        std_l2, std_o, ansi("OK", A.GREEN)) if memory_ok else
        "   std(l2)={:.4f}  std(o)={:.4f}  {} prova piu' episodi o riduci O_BETA".format(
        std_l2, std_o, ansi("--", A.YELLOW)))
    print(mem_msg)

    # distribuzione stati
    print()
    print("  Distribuzione stati (ultimi {} ep):".format(last))
    print("    {:<20s} {:>4d}  {}".format(
        ansi("CRISTALLIZZATO", A.CYAN),   n_cryst, bar_u(n_cryst / max(last, 1), 20)))
    print("    {:<20s} {:>4d}  {}".format(
        ansi("CRITICO",        A.YELLOW), n_crit,  bar_u(n_crit  / max(last, 1), 20)))
    print("    {:<20s} {:>4d}  {}".format(
        ansi("INSTABILE",      A.RED),    n_inst,  bar_u(n_inst  / max(last, 1), 20)))

    # verdetti
    print()
    lyap_v = ansi("VERIFICATO OK",     A.GREEN)  if lyap_g   > 0.55 else ansi("PARZIALE ~",      A.YELLOW)
    rri_v  = ansi("REGIMI STABILI OK", A.GREEN)  if rri_last > 0.10 else ansi("FLUIDO ~",         A.YELLOW)
    mem_v  = ansi("MEMORIA OK",        A.GREEN)  if memory_ok        else ansi("ANCORA FLUIDO ~", A.YELLOW)
    if   n_g >= 3: plu_v = ansi("PLURALISMO STRUTTURATO OK", A.CYAN)
    elif n_g >= 2: plu_v = ansi("COESISTENZA ~",             A.YELLOW)
    else:          plu_v = ansi("MONO-REGIME !",              A.RED)
    l2_v = (ansi("CRISTALLIZZATO", A.CYAN)   if l2_last > 0.65
            else ansi("CRITICO",    A.YELLOW) if l2_last > 0.40
            else ansi("INSTABILE",  A.RED))
    print("  Lyapunov ({:.1f}%):  {}".format(lyap_g * 100, lyap_v))
    print("  RRI ({:.3f}):        {}".format(rri_last, rri_v))
    print("  Pluralismo ({:.1f}): {}".format(n_g, plu_v))
    print("  Stato finale l2:    {}".format(l2_v))
    print("  Memoria (o vs l2):  {}".format(mem_v))

    # agenti
    print()
    print(ansi("  AGENTI POST-TRAINING", A.BOLD, A.WHITE))
    rl, sizes, centroids, n_reg, dom_idx, dom_frac = rd.detect(agents)
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    for i, a in enumerate(agents):
        lbl = int(rl[i]) if i < len(rl) else -1
        ch  = chars[lbl % len(chars)] if lbl >= 0 else "?"
        tn  = float(np.linalg.norm(a.tau))
        bc  = A.CYAN if a._ali_score > a._ant_score else A.YELLOW
        print("  Agent {:>2d} [R:{}]  tau={}  {}  curv={:.4f}  ali={}  ant={}".format(
            a.uid,
            ansi(ch, bc),
            ansi("{:.3f}".format(tn), A.WHITE),
            ansi(bar_u(tn / 1.2, 10), bc),
            a.tau_curv,
            ansi("{:.3f}".format(a._ali_score), A.GREEN),
            ansi("{:.3f}".format(a._ant_score), A.RED),
        ))

    if len(centroids) >= 2:
        print()
        print(ansi("  SEPARAZIONE REGIMI (coseno)", A.DIM))
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                cs  = float(np.dot(centroids[i], centroids[j]) /
                            (np.linalg.norm(centroids[i]) * np.linalg.norm(centroids[j]) + 1e-8))
                sep = 1 - cs
                sc  = A.CYAN if sep > 0.5 else A.YELLOW if sep > 0.25 else A.RED
                print("    R{}-R{}: cos={:.3f}  sep={}".format(
                    i, j, cs, ansi("{:.3f}".format(sep), sc)))


# ===============================================================
# PLOT
# ===============================================================
def plot_results(history, out_path="symphonon_v17_results.png"):
    if not HAS_PLT:
        print(ansi("  [INFO] matplotlib non disponibile -- grafici saltati.", A.DIM))
        return

    fig = plt.figure(figsize=(15, 13), facecolor="#06080f")
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.50, wspace=0.35)

    def ax_style(ax, title, ylabel=""):
        ax.set_facecolor("#0d1220")
        ax.tick_params(colors="#4a6080", labelsize=8)
        ax.set_title(title, color="#c8daf0", fontsize=9, pad=6)
        ax.set_ylabel(ylabel, color="#4a6080", fontsize=8)
        for spine in ax.spines.values():
            spine.set_color("#1a2540")
        ax.grid(True, color="#1a2540", linewidth=0.5)

    ep = np.arange(len(history['F']))

    # riga 1
    ax = fig.add_subplot(gs[0, :2])
    ax_style(ax, "lambda2 -- Coherence Index (indicatore istantaneo)", "lambda2")
    ax.plot(ep, history['lambda2'], color="#00e5cc", lw=1.2, label="lambda2")
    ax.axhline(0.65, color="#00e5cc", lw=0.6, ls="--", alpha=0.5, label="cristallo 0.65")
    ax.axhline(0.40, color="#ffa520", lw=0.6, ls="--", alpha=0.5, label="critico 0.40")
    ax.fill_between(ep, history['lambda2'], 0.65,
                    where=[v >= 0.65 for v in history['lambda2']],
                    color="#00e5cc", alpha=0.08)
    ax.fill_between(ep, history['lambda2'], 0.40,
                    where=[0.40 <= v < 0.65 for v in history['lambda2']],
                    color="#ffa520", alpha=0.08)
    ax.fill_between(ep, history['lambda2'], 0,
                    where=[v < 0.40 for v in history['lambda2']],
                    color="#ff3c4e", alpha=0.10)
    ax.legend(fontsize=7, facecolor="#0d1220", edgecolor="#1a2540", labelcolor="#c8daf0")
    ax.set_ylim(0, 1)

    ax2 = fig.add_subplot(gs[0, 2])
    ax_style(ax2, "Stato (distribuzione)")
    state_map = {"CRISTALLIZZATO": 2, "CRITICO": 1, "INSTABILE": 0}
    cols      = {2: "#00e5cc", 1: "#ffa520", 0: "#ff3c4e"}
    counts    = {s: history['state'].count(s) for s in ["CRISTALLIZZATO", "CRITICO", "INSTABILE"]}
    bplt      = ax2.bar(counts.keys(), counts.values(),
                        color=[cols[state_map[s]] for s in counts], width=0.5)
    ax2.tick_params(axis='x', labelrotation=20, labelsize=7)
    for b, (s, c) in zip(bplt, counts.items()):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 5,
                 str(c), ha='center', va='bottom', color="#c8daf0", fontsize=8)

    # riga 2
    ax3 = fig.add_subplot(gs[1, 0])
    ax_style(ax3, "RRI & GFC", "valore")
    ax3.plot(ep, history['rri'], color="#00e5cc", lw=1, label="RRI")
    ax3.plot(ep, history['gfc'], color="#a78bfa", lw=1, label="GFC")
    ax3.legend(fontsize=7, facecolor="#0d1220", edgecolor="#1a2540", labelcolor="#c8daf0")

    ax4 = fig.add_subplot(gs[1, 1])
    ax_style(ax4, "Psi -- Dominanza (basso=buono)", "Psi")
    ax4.plot(ep, history['psi'], color="#ff3c4e", lw=1)
    ax4.axhline(0.70, color="#ff3c4e", lw=0.6, ls="--", alpha=0.4, label="soglia collasso")
    ax4.legend(fontsize=7, facecolor="#0d1220", edgecolor="#1a2540", labelcolor="#c8daf0")

    ax5 = fig.add_subplot(gs[1, 2])
    ax_style(ax5, "F -- Free Energy", "F")
    ax5.plot(ep, history['F'], color="#4a90d9", lw=1)
    ax5.axhline(0, color="#4a6080", lw=0.5, ls="--")

    # riga 3
    ax6 = fig.add_subplot(gs[2, 0])
    ax_style(ax6, "Reward medio", "R")
    ax6.plot(ep, history['reward'], color="#80ff80", lw=1)
    ax6.axhline(0, color="#4a6080", lw=0.5, ls="--")

    ax7 = fig.add_subplot(gs[2, 1])
    ax_style(ax7, "N regimi distinti", "n")
    ax7.plot(ep, history['n_regimes'], color="#a78bfa", lw=1, drawstyle="steps-post")
    ax7.axhline(2, color="#a78bfa", lw=0.5, ls="--", alpha=0.4)

    ax8 = fig.add_subplot(gs[2, 2])
    ax_style(ax8, "Lyapunov %  (dF<0)", "%")
    smooth = np.convolve(history['lyap'], np.ones(20) / 20, mode='same')
    ax8.plot(ep, history['lyap'], color="#4a6080", lw=0.5, alpha=0.4)
    ax8.plot(ep, smooth, color="#00e5cc", lw=1.2, label="media mobile")
    ax8.axhline(0.55, color="#00e5cc", lw=0.6, ls="--", alpha=0.5, label="soglia OK 55%")
    ax8.legend(fontsize=7, facecolor="#0d1220", edgecolor="#1a2540", labelcolor="#c8daf0")
    ax8.set_ylim(0, 1)

    # riga 4: PARAMETRO D'ORDINE (novita' v17)
    ax9 = fig.add_subplot(gs[3, :2])
    ax_style(ax9, "lambda2 vs o -- indicatore istantaneo vs attrattore dinamico", "valore")
    ax9.plot(ep, history['lambda2'], color="#00e5cc", lw=1.0, alpha=0.55, label="lambda2 (istantaneo)")
    ax9.plot(ep, history['o'],       color="#ff9f1c", lw=1.6,             label="o  (attrattore)")
    ax9.fill_between(ep, history['lambda2'], history['o'],
                     alpha=0.08, color="#ff9f1c", label="|tensione|")
    ax9.axhline(0.65, color="#00e5cc", lw=0.4, ls="--", alpha=0.3)
    ax9.axhline(0.40, color="#ffa520", lw=0.4, ls="--", alpha=0.3)
    ax9.legend(fontsize=7, facecolor="#0d1220", edgecolor="#1a2540", labelcolor="#c8daf0")
    ax9.set_ylim(0, 1)

    ax10 = fig.add_subplot(gs[3, 2])
    ax_style(ax10, "stabilita' di o  (1-std)", "o_stab")
    ax10.plot(ep, history['o_stability'], color="#ff9f1c", lw=1)
    ax10.axhline(0.80, color="#ff9f1c", lw=0.6, ls="--", alpha=0.5, label="attrattore 0.80")
    ax10.legend(fontsize=7, facecolor="#0d1220", edgecolor="#1a2540", labelcolor="#c8daf0")
    ax10.set_ylim(0, 1)

    fig.suptitle("SYMPHONON O v17 -- Coherence Probe + Parametro d'Ordine",
                 color="#00e5cc", fontsize=13, fontweight="bold", y=0.99)

    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="#06080f")
    plt.close(fig)
    print(ansi("  Grafici salvati --> {}".format(out_path), A.GREEN))


# ===============================================================
# PROBE MODE (CLI interattivo)
# ===============================================================
def probe_mode(judge=None):
    if judge is None:
        judge = CoherenceJudge()

    W = 62
    print()
    print(ansi("-" * W, A.CYAN))
    print(ansi("  PROBE MODE -- inserisci valori, ottieni la diagnosi", A.BOLD, A.CYAN))
    print(ansi("  comandi: preset  |  4 numeri  |  reset  |  q", A.DIM))
    print(ansi("-" * W, A.CYAN))
    print(ansi("  Parametri: psi  rri  gfc  reward  (tutti in [0,1])", A.DIM))
    print()

    presets = {
        "cristallo": (0.10, 0.75, 0.40, 0.80),
        "critico":   (0.45, 0.45, 0.25, 0.40),
        "caos":      (0.85, 0.05, 0.08, 0.05),
    }
    print(ansi("  Preset:", A.DIM))
    for k, (ps, ri, gf, rw) in presets.items():
        jd = judge.evaluate(ps, ri, gf, rw)
        print("    {:<10s}  psi={}  rri={}  gfc={}  rew={}  ->  {}".format(
            k, ps, ri, gf, rw, ansi(jd['state'], jd['color'])))
    print()

    while True:
        try:
            raw = input(ansi("  > ", A.CYAN)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue
        if raw.lower() in ('q', 'quit', 'exit'):
            break
        if raw.lower() == 'reset':
            judge.reset()
            print(ansi("  Storia e o azzerati.", A.DIM))
            continue

        if raw.lower() in presets:
            ps, ri, gf, rw = presets[raw.lower()]
        else:
            parts = raw.replace("=", " ").split()
            nums  = []
            for p in parts:
                try:
                    nums.append(float(p))
                except ValueError:
                    pass
            if len(nums) < 4:
                print(ansi("  Formato: psi rri gfc reward  (es. 0.15 0.70 0.35 0.75)", A.RED))
                continue
            ps, ri, gf, rw = nums[:4]

        try:
            jd = judge.evaluate(ps, ri, gf, rw)
        except Exception as e:
            print(ansi("  Errore: {}".format(e), A.RED))
            continue

        l2_s    = "{:.4f}".format(jd['lambda2'])
        o_s     = "{:.4f}".format(jd['o'])
        ten_s   = "{:+.4f}".format(jd['tension'])
        o_col   = A.CYAN if jd['o'] > 0.65 else A.YELLOW if jd['o'] > 0.40 else A.RED
        ten_col = A.GREEN if jd['tension'] > 0.02 else A.RED if jd['tension'] < -0.02 else A.DIM
        ten_lbl = ("inerzia -- memoria alta" if jd['tension'] >  0.05
                   else "l2 salito, o segue"   if jd['tension'] < -0.05
                   else "equilibrio")
        stab_lbl = "attrattore raggiunto" if jd['o_stability'] > 0.8 else "in transizione"

        print()
        print("  {} {}".format(jd['icon'], ansi(jd['state'], jd['color'], A.BOLD)))
        print("     l2      = {}  [{}]".format(
            ansi(l2_s, jd['color']), ansi(bar_u(jd['lambda2'], 22), jd['color'])))
        print("     o       = {}  [{}]  <- attrattore".format(
            ansi(o_s, o_col), ansi(bar_u(jd['o'], 22), o_col)))
        print("     tension = {}  {}".format(ansi(ten_s, ten_col), ten_lbl))
        print("     o_stab  = {:.4f}  {}".format(jd['o_stability'], stab_lbl))
        print("     RRI_J   = {:.4f}".format(jd['rri_j']))
        print("     drift   = {:.4f}  {}".format(
            jd['drift'], "[!]" if jd['drift'] > 0.1 else ""))
        print("     trend   = {:>+.5f}  {}".format(
            jd['trend'], ansi(jd['forecast'], A.DIM)))
        print("     {}".format(ansi(jd['msg'], A.DIM)))
        print()


# ===============================================================
# MAIN
# ===============================================================
def main():
    parser = argparse.ArgumentParser(description="Symphonon O v17 -- Coherence Probe")
    parser.add_argument("--episodes", type=int,  default=None, help="Numero episodi (default: 1200)")
    parser.add_argument("--agents",   type=int,  default=None, help="Numero agenti  (default: 12)")
    parser.add_argument("--seed",     type=int,  default=42,   help="Random seed")
    parser.add_argument("--probe",    action="store_true",     help="Solo probe CLI, salta training")
    parser.add_argument("--no-plot",  action="store_true",     help="Non salvare grafici")
    args = parser.parse_args()

    if args.probe:
        probe_mode()
        return

    agents, history, judge = train(
        n_episodes = args.episodes,
        n_agents   = args.agents,
        seed       = args.seed,
    )

    summary(history, agents)

    if not args.no_plot:
        plot_results(history)

    print()
    print(ansi("  Avvio PROBE interattivo...", A.DIM))
    probe_mode(judge)

    print()
    print(ansi("  Fine.", A.CYAN))


if __name__ == "__main__":
    main()
