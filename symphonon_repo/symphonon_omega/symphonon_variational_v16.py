"""
SYMPHONON - Variational Multi-Agent Self-Organization v7
=========================================================
Closed crystal-melt-crystal cycle via phase memory, recrystallization trigger,
and smooth locking. All previous fixes incorporated cleanly.

Version history compressed:
  v1: baseline Langevin on homogeneous grid
  v2: param fixes (F_ALPHA, F_LAMBDA, noise gate, EMA faster, SOC_THR lower)
  v3: heterogeneous patchy environment + directional Laplacian + competition
  v4: noise gate (D_eff = 0.25 + psi*scale), inertia (0.92*tau + 0.08*tau_old),
      stronger repulsion (0.8)
  v5: TAU_CLIP=1.2, F_ALPHA=0.10 (tau commitment)
  v6: time dilation (conditional locking lock=0.3 when stable)
  v7: [A] phase memory (tau_anchor, pull toward past crystal)
      [B] recrystallization trigger (halve noise after 80 low-RRI episodes)
      [C] smooth locking (sigmoid on tau_curv instead of hard threshold)

Run:   python symphonon_v7.py
Deps:  numpy (PyTorch optional, activates policy gradient if installed)
"""

import numpy as np
import sys
import time
from collections import deque, defaultdict

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F_torch
    TORCH = True
except ImportError:
    TORCH = False
    print("[INFO] PyTorch not available - numpy-only mode")


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
class Config:
    N_AGENTS        = 12
    TAU_DIM         = 16
    OBS_DIM         = 20
    ACTION_DIM      = 5
    HIDDEN_DIM      = 64

    # Free energy functional weights
    F_ALPHA         = 0.10   # v5: push away from mean (tau commitment)
    F_BETA          = 0.04
    F_GAMMA         = 0.08
    F_DELTA         = 0.06
    F_LAMBDA        = 0.25   # v2: task guides field

    # Langevin
    LANGEVIN_DT     = 0.015
    D_BASE          = 0.010
    D_PSI_SCALE     = 0.80   # v2: less random reset

    # Tau constraints
    TAU_CLIP        = 1.20   # v5: more room for strong identities
    TAU_FLOOR       = 0.015

    # Social graph
    SOC_SCORE_THR   = 0.02   # v2: lower -> more active connections
    EMA_LONG        = 0.050  # v2: 3x faster
    EMA_SHORT       = 0.150  # v2: 2x faster
    SCORE_LONG_W    = 0.65

    # Regime detection
    REGIME_COS_THR  = 0.35   # v3: easier cluster formation
    REGIME_MIN_SIZE = 2

    # Phase memory (v7)
    ANCHOR_RRI_THR  = 0.40   # RRI above this -> save anchor
    ANCHOR_PULL_THR = 0.30   # RRI below this -> apply pull
    ANCHOR_BETA     = 0.04   # [v9] slightly stronger pull for nucleation

    # Recrystallization trigger (v7)
    LOW_RRI_THR     = 0.05   # RRI threshold for counting
    LOW_RRI_K       = 80     # episodes before trigger
    RECRYSTAL_DUR   = 120    # episodes of reduced noise
    RECRYSTAL_FACT  = 0.50   # noise reduction factor

    # Smooth locking (v7)
    LOCK_CURV_MID   = 0.12   # sigmoid midpoint
    LOCK_CURV_SCALE = 0.03   # sigmoid sharpness
    LOCK_MIN        = 0.30   # minimum lock (fully stable)
    LOCK_ALI_BOOST  = 0.70   # extra reduction when ali > ant

    # Multi-anchor memory (v8)
    ANCHOR_DECAY    = 0.9980   # anchors fade slowly (prevent fossilization)
    ANCHOR_MAX      = 8        # max number of historical anchors kept
    ANCHOR_COMPETE_W = 0.025   # pull strength toward own-regime anchor
    ANCHOR_REPEL_W  = 0.008   # repulsion from other-regime anchors

    # Policy training (v15: linear policy + tau bias)
    LR_POLICY       = 3e-3   # higher lr for linear policy
    TAU_POLICY_ALPHA = 5e-4  # [v16] lower base rate, further gated by Psi
    GAMMA_RL        = 0.97
    ENTROPY_COEF    = 0.01
    TAU_ALIGN_W     = 0.30

    # Training
    N_EPISODES      = 1200
    EPISODE_LEN     = 60
    LOG_EVERY       = 40
    LYAP_WINDOW     = 50

    # Environment
    GRID_SIZE       = 10
    FOOD_REWARD     = 1.0
    MOVE_COST       = 0.01


C = Config()


# ---------------------------------------------------------------------------
# POLICY NETWORK
# ---------------------------------------------------------------------------
class LinearPolicy:
    """
    Minimal linear policy: theta (OBS_DIM x ACTION_DIM matrix).

    Forward: logits = obs @ theta + tau_bias
    tau_bias: tau projected into action space as a structural bias.
    tau influences WHICH actions are preferred, not just how much.

    Update (REINFORCE):
        theta += lr * reward * grad_log_pi(a | obs)
        theta += alpha * tau_projection  (structural bias from trajectory)

    This closes the loop:
        tau -> policy bias -> action -> reward -> policy update -> tau
    """
    def __init__(self):
        self.theta      = np.random.randn(C.OBS_DIM, C.ACTION_DIM).astype(np.float32) * 0.05
        # Projection matrix: TAU_DIM -> ACTION_DIM (fixed random, learned indirectly)
        self.tau_proj   = np.random.randn(C.TAU_DIM, C.ACTION_DIM).astype(np.float32) * 0.02
        # Accumulated eligibility trace for REINFORCE
        self._last_obs  = None
        self._last_act  = -1
        self._last_probs = None

    def _forward(self, obs, tau):
        """logits = obs @ theta + tau @ tau_proj"""
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
        """
        REINFORCE with baseline + tau structural bias.

        Baseline reduces variance: advantage = reward - mean_reward.
        TAU_POLICY_ALPHA is gated by Psi: when system collapses (psi high),
        tau influence on policy is reduced to prevent synchronization cascade.
        """
        if self._last_obs is None: return
        lr    = C.LR_POLICY
        obs   = self._last_obs
        a     = self._last_act
        probs = self._last_probs

        # REINFORCE with baseline
        advantage = float(reward) - float(mean_reward)
        grad_log  = -probs.copy()
        grad_log[a] += 1.0
        self.theta += lr * advantage * np.outer(obs, grad_log)

        # Tau structural bias gated by Psi:
        # when Psi high (collapse risk), reduce coupling tau->policy
        psi_gate = 1.0 - 1.0 / (1.0 + np.exp(-(float(psi) - 0.30) / 0.08))
        alpha    = C.TAU_POLICY_ALPHA * psi_gate
        tau_signal = tau @ self.tau_proj
        tn = float(np.linalg.norm(tau_signal)) + 1e-8
        self.theta += alpha * np.outer(obs, tau_signal / tn)

        np.clip(self.theta, -2.0, 2.0, out=self.theta)


# Use LinearPolicy regardless of torch availability
PolicyNet = LinearPolicy


# ---------------------------------------------------------------------------
# AGENT
# ---------------------------------------------------------------------------
class Agent:
    _uid_counter = 0

    def __init__(self):
        self.uid         = Agent._uid_counter
        Agent._uid_counter += 1
        self.policy      = PolicyNet()
        self.tau         = np.random.randn(C.TAU_DIM).astype(np.float32) * 0.3
        self._tau_enforce()
        self.social_mem  = {}
        self._ali_score  = 0.0
        self._ant_score  = 0.0
        self.tau_curv    = 0.02
        self.regime_idx  = -1
        self.ep_log_probs  = []
        self.ep_rewards    = []
        self.ep_entropies  = []
        self.ep_tau_aligns = []
        # LinearPolicy has no separate optimizer

    def _tau_enforce(self):
        n = float(np.linalg.norm(self.tau))
        if   n > C.TAU_CLIP:           self.tau *= C.TAU_CLIP / n
        elif 1e-9 < n < C.TAU_FLOOR:   self.tau *= C.TAU_FLOOR / n
        elif n <= 1e-9:                 self.tau  = np.random.randn(C.TAU_DIM).astype(np.float32) * C.TAU_FLOOR

    def relation_score(self, uid):
        if uid not in self.social_mem: return 0.0
        l, s = self.social_mem[uid]
        return l * C.SCORE_LONG_W + s * (1.0 - C.SCORE_LONG_W)

    def update_social(self, other_uid, outcome):
        prev  = self.social_mem.get(other_uid, [0.0, 0.0])
        ln    = prev[0] * (1 - C.EMA_LONG)  + outcome * C.EMA_LONG
        sn    = prev[1] * (1 - C.EMA_SHORT) + outcome * C.EMA_SHORT
        self.social_mem[other_uid] = [ln, sn]

    def reset_episode(self):
        self.ep_log_probs  = []
        self.ep_rewards    = []
        self.ep_entropies  = []
        self.ep_tau_aligns = []


# ---------------------------------------------------------------------------
# REGIME DETECTOR
# ---------------------------------------------------------------------------
class RegimeDetector:

    @staticmethod
    def detect(agents):
        n = len(agents)
        if n == 0:
            return np.array([], dtype=int), np.array([]), np.zeros((0, C.TAU_DIM)), 0, -1, 0.0
        taus     = np.array([a.tau for a in agents], dtype=np.float32)
        tau_hat  = taus / (np.linalg.norm(taus, axis=1, keepdims=True) + 1e-8)
        labels   = np.full(n, -1, dtype=int)
        centroids = []; cent_norm = []
        for i in range(n):
            if not centroids:
                labels[i] = 0; centroids.append(tau_hat[i].copy()); cent_norm.append(tau_hat[i].copy()); continue
            sims = np.array([float(np.dot(tau_hat[i], c)) for c in cent_norm])
            best = int(np.argmax(sims))
            if sims[best] >= C.REGIME_COS_THR:
                labels[i] = best; k = int(np.sum(labels == best))
                centroids[best] = (centroids[best] * (k-1) + tau_hat[i]) / k
                cn = float(np.linalg.norm(centroids[best])); cent_norm[best] = centroids[best] / (cn + 1e-8)
            else:
                nid = len(centroids); labels[i] = nid
                centroids.append(tau_hat[i].copy()); cent_norm.append(tau_hat[i].copy())
        nc    = len(centroids)
        sizes = np.array([int(np.sum(labels == k)) for k in range(nc)])
        sig   = np.where(sizes >= C.REGIME_MIN_SIZE)[0]; nr = len(sig)
        if nr == 0: return labels, sizes, np.array(cent_norm), 0, -1, 0.0
        di = int(sig[np.argmax(sizes[sig])]); df = float(sizes[di]) / n
        return labels, sizes, np.array(cent_norm), nr, di, df

    @staticmethod
    def compute_psi(sizes, n_total):
        sig = sizes[sizes >= C.REGIME_MIN_SIZE]; K = len(sig)
        if K <= 1 or n_total == 0: return 1.0 if K <= 1 else 0.0
        p = sig / sig.sum(); H = -np.sum(p * np.log(p + 1e-12)); Hmax = np.log(K)
        return float(1.0 - H / Hmax) if Hmax > 0 else 1.0

    @staticmethod
    def compute_rri(agents, labels, dom_idx):
        if dom_idx < 0 or not agents: return 0.0
        members = [a for i, a in enumerate(agents) if i < len(labels) and labels[i] == dom_idx]
        if not members: return 0.0
        return float(np.clip(1.0 - np.mean([m.tau_curv for m in members]) / (C.TAU_FLOOR * 4), 0.0, 1.0))

    @staticmethod
    def compute_gfc(taus):
        if len(taus) < 3: return 0.0
        tn = taus / (np.linalg.norm(taus, axis=1, keepdims=True) + 1e-8)
        return float(np.sqrt(np.mean(tn[:, 0])**2 + np.mean(tn[:, 1])**2))

    @staticmethod
    def compute_psi_p(agents):
        if len(agents) < 2: return 0.5
        taus = np.array([a.tau for a in agents], dtype=np.float32)
        tn   = taus / (np.linalg.norm(taus, axis=1, keepdims=True) + 1e-8)
        n    = min(len(agents), 8)
        idx  = np.random.choice(len(agents), n, replace=False)
        cv   = [float(np.dot(tn[idx[i]], tn[idx[j]])) for i in range(n) for j in range(i+1, n)]
        return float((1.0 - np.mean(cv)) * 0.5) if cv else 0.5


# ---------------------------------------------------------------------------
# VARIATIONAL FIELD  (v7: phase memory + recrystal trigger + smooth locking)
# ---------------------------------------------------------------------------
class VariationalField:
    # Class-level phase state (persists across episodes)
    # v8: per-regime anchors replace single global anchor
    _anchors       = {}     # regime_idx -> tau_anchor (decaying historical echo)
    _low_rri_count = 0
    _recrystal_on  = False
    _recrystal_cnt = 0

    @staticmethod
    def _softmax(v):
        e = np.exp(v - v.max()); return e / (e.sum() + 1e-8)

    @staticmethod
    def compute_F(psi, rri, gfc, psi_p, reward_norm=0.0):
        return (C.F_ALPHA * psi + C.F_BETA * rri
                - C.F_GAMMA * gfc - C.F_DELTA * psi_p
                - C.F_LAMBDA * reward_norm)

    @staticmethod
    def local_gradients(agent, agents, mu_pop):
        tau_i = agent.tau

        # [A] Anti-collapse: push away from population mean
        dPsi = tau_i - mu_pop

        # [B] Anti-dominance: penalise strongest axis in tau
        sw   = VariationalField._softmax(np.abs(tau_i))
        dRRI = sw * np.sign(tau_i + 1e-9)

        # [C] Directional social Laplacian: allies attract, antagonists repel
        lap = np.zeros(C.TAU_DIM, dtype=np.float32); n_nb = 0
        for other in agents:
            if other.uid == agent.uid: continue
            sc = agent.relation_score(other.uid)
            if abs(sc) > C.SOC_SCORE_THR:
                if sc > 0:
                    lap += (other.tau - tau_i) * sc        # ally: attract
                else:
                    lap -= (other.tau - tau_i) * abs(sc) * 0.8   # antagonist: repel (v4)
                n_nb += 1
        if n_nb > 0: lap /= n_nb
        dG = -lap

        # [D] Internal entropy: distribute tau mass across axes
        p         = VariationalField._softmax(np.abs(tau_i))
        dPsiLocal = -(np.log(p + 1e-9) + 1.0) * np.sign(tau_i + 1e-9)

        return (C.F_ALPHA * dPsi + C.F_BETA * dRRI
                + C.F_GAMMA * dG + C.F_DELTA * dPsiLocal)

    @staticmethod
    def langevin_step(agents, psi, rri, gfc, psi_p, reward_norm=0.0):
        """
        Euler-Maruyama on tau - v8: multi-anchor ecosystem.

        [A] Per-regime anchors:
            Each regime gets its own decaying historical echo.
            anchor[k] = EMA of mean_tau for regime k when RRI > threshold.
            All anchors decay by ANCHOR_DECAY each episode (prevent fossilization).

        [B] Competing pulls:
            Each agent is pulled toward its own regime anchor (ANCHOR_COMPETE_W)
            and weakly repelled from other-regime anchors (ANCHOR_REPEL_W).
            Result: ecological memory - regimes remember their own history,
            not a single global attractor.

        [C] Recrystallization trigger (unchanged from v7).

        [D] Smooth locking (unchanged from v7).
        """
        if not agents: return

        vf     = VariationalField
        mu_pop = np.mean([a.tau for a in agents], axis=0)

        # Compute social scores
        for agent in agents:
            ali_a = ant_a = ali_c = ant_c = 0
            for other in agents:
                if other.uid == agent.uid: continue
                sc = agent.relation_score(other.uid)
                if   sc > 0: ali_a += sc; ali_c += 1
                elif sc < 0: ant_a += -sc; ant_c += 1
            agent._ali_score = ali_a / max(ali_c, 1)
            agent._ant_score = ant_a / max(ant_c, 1)

        # [A] Update per-regime anchors
        # Group agents by regime
        regime_taus = {}
        for agent in agents:
            rid = agent.regime_idx
            if rid < 0: continue
            if rid not in regime_taus: regime_taus[rid] = []
            regime_taus[rid].append(agent.tau)

        # [v14] Anchor update fires at lower RRI threshold (0.25)
        # and EMA rate scales with RRI: brief cycles contribute proportionally
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

        # [v12] State-dependent anchor decay:
        # crystal phase (rri > 0.3) -> anchor nearly stable
        # liquid phase -> anchor decays normally
        decay_eff = C.ANCHOR_DECAY + (1.0 - C.ANCHOR_DECAY) * (
            1.0 / (1.0 + np.exp(-(rri - 0.30) / 0.08)))
        for rid in list(vf._anchors.keys()):
            vf._anchors[rid] *= decay_eff
            if float(np.linalg.norm(vf._anchors[rid])) < C.TAU_FLOOR:
                del vf._anchors[rid]

        # Prune to ANCHOR_MAX (keep most recent / strongest)
        if len(vf._anchors) > C.ANCHOR_MAX:
            norms = {rid: float(np.linalg.norm(a)) for rid, a in vf._anchors.items()}
            keep  = sorted(norms, key=norms.get, reverse=True)[:C.ANCHOR_MAX]
            vf._anchors = {rid: vf._anchors[rid] for rid in keep}

        # [v13] Continuous recrystallization: no discrete trigger, no dead zone
        # As soon as RRI drops, noise starts reducing immediately.
        # sigmoid((0.15 - rri) / 0.05) -> 1 when rri~0, 0 when rri>0.3
        recryst_strength = 1.0 / (1.0 + np.exp(-(0.15 - rri) / 0.05))
        r_factor = 1.0 - 0.60 * recryst_strength
        # Keep counters for logging but they no longer gate the mechanism
        if rri < C.LOW_RRI_THR:
            vf._low_rri_count += 1
            vf._recrystal_on = (vf._low_rri_count > 5)
        else:
            vf._low_rri_count = max(0, vf._low_rri_count - 1)
            vf._recrystal_on = False

        D_eff     = C.D_BASE * (0.25 + float(psi) * C.D_PSI_SCALE) * r_factor
        # [v11] Cycle inertia: crystal phase is more viscous
        # sigmoid(rri - 0.4) -> 0 in liquid, ~0.5..1 in crystal
        cycle_inertia = 1.0 / (1.0 + np.exp(-(rri - 0.40) / 0.08))
        D_eff    *= (1.0 - 0.35 * cycle_inertia)
        noise_std = float(np.sqrt(2.0 * D_eff * C.LANGEVIN_DT))

        for agent in agents:
            tau_old = agent.tau.copy()
            grad_F  = VariationalField.local_gradients(agent, agents, mu_pop)

            # Task surplus
            last_r  = float(agent.ep_rewards[-1]) if agent.ep_rewards else 0.0
            surplus = float(np.clip(last_r - reward_norm, -0.5, 0.5))
            if abs(surplus) > 0.005:
                tau_dir = tau_old / (float(np.linalg.norm(tau_old)) + 1e-8)
                grad_F  = grad_F - tau_dir * surplus * C.F_LAMBDA * 0.5

            # [B] Competing anchors: own-regime pull + other-regime repulsion
            # [v10] Nucleation gate: when RRI < 0.20, pull is boosted 8x
            # and noise is reduced by a separate factor applied below.
            # This breaks the chicken-and-egg: noise prevents locking,
            # locking prevents crystallization.
            nucl_gate = 1.0 / (1.0 + np.exp((rri - 0.15) / 0.04))  # 1 when rri<0.15
            pull_boost = 1.0 + 7.0 * nucl_gate   # up to 8x stronger pull
            own_rid = agent.regime_idx
            for rid, anchor in vf._anchors.items():
                if rid == own_rid:
                    pull    = C.ANCHOR_COMPETE_W * pull_boost * (anchor - agent.tau)
                    grad_F  = grad_F - pull
                else:
                    repel_gate = 1.0 / (1.0 + np.exp(-(rri - 0.20) / 0.06))
                    diff    = agent.tau - anchor
                    dn      = float(np.linalg.norm(diff)) + 1e-8
                    repel   = C.ANCHOR_REPEL_W * repel_gate * (diff / dn)
                    grad_F  = grad_F + repel

            # [D] Smooth locking (sigmoid on tau_curv)
            curv_dist = (C.LOCK_CURV_MID - agent.tau_curv) / C.LOCK_CURV_SCALE
            sig       = 1.0 / (1.0 + np.exp(-curv_dist))
            lock      = C.LOCK_MIN + (1.0 - C.LOCK_MIN) * (1.0 - sig)
            if agent._ali_score > agent._ant_score + 0.01:
                lock *= C.LOCK_ALI_BOOST

            # [v10] Extra noise reduction during nucleation (beyond recryst)
            noise_scale = lock * (1.0 - 0.65 * nucl_gate)  # up to 65% less noise
            noise     = np.random.randn(C.TAU_DIM).astype(np.float32) * noise_std * noise_scale
            agent.tau = tau_old - lock * C.LANGEVIN_DT * grad_F + noise
            agent.tau = 0.92 * agent.tau + 0.08 * tau_old
            agent._tau_enforce()

            to_n = float(np.linalg.norm(tau_old)); tn_n = float(np.linalg.norm(agent.tau))
            if to_n > 1e-6 and tn_n > 1e-6:
                raw_c = float(np.linalg.norm(agent.tau / tn_n - tau_old / to_n))
            else:
                raw_c = 0.0
            agent.tau_curv = agent.tau_curv * 0.96 + raw_c * 0.04


# ---------------------------------------------------------------------------
# ENVIRONMENT - patchy heterogeneous foraging grid (v3)
# ---------------------------------------------------------------------------
class ForagingGrid:
    N_PATCHES     = 4
    PATCH_SIZE    = 2
    PATCH_REWARDS = [1.0, 0.8, 1.2, 0.9]
    RESPAWN_RATE  = 0.15

    def __init__(self):
        self.G = C.GRID_SIZE
        g = self.G
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
            if (int(x), int(y)) in pset: return p
        return -1

    def reset(self):
        g = self.G
        self.pos = np.zeros((C.N_AGENTS, 2), dtype=int)
        for i in range(C.N_AGENTS):
            p = i % self.N_PATCHES; cx, cy = self.patch_centers[p]
            self.pos[i] = [(cx + np.random.randint(-1, 2)) % g,
                           (cy + np.random.randint(-1, 2)) % g]
        self.food = np.zeros((g, g), dtype=np.float32)
        for p, cells in enumerate(self.patch_cells):
            rv = self.PATCH_REWARDS[p]
            for cx, cy in cells:
                if np.random.rand() < 0.7: self.food[cx, cy] = rv
        self.reward_ema = np.zeros(C.N_AGENTS)
        self.prev_pos   = self.pos.copy()

    def _obs(self, i):
        x, y = self.pos[i]; g = self.G
        obs = np.zeros(C.OBS_DIM, dtype=np.float32)
        obs[0] = x / g; obs[1] = y / g
        obs[2] = float((x - self.prev_pos[i, 0]) / g)
        obs[3] = float((y - self.prev_pos[i, 1]) / g)
        dists = [float(np.linalg.norm(self.patch_centers[p] - np.array([x, y])) / g)
                 for p in range(self.N_PATCHES)]
        sp = np.argsort(dists)
        obs[4] = dists[sp[0]]; obs[5] = float(self.patch_centers[sp[0], 0] - x) / g
        obs[6] = dists[sp[1]]; obs[7] = float(self.patch_centers[sp[1], 0] - x) / g
        cp = self._cell_patch(x, y)
        if 0 <= cp < 4: obs[8 + cp] = 1.0
        n_near = sum(1 for j in range(C.N_AGENTS) if j != i and
                     np.linalg.norm(self.pos[j] - self.pos[i]) < 2.5)
        obs[12] = n_near / C.N_AGENTS; obs[13] = float(self.reward_ema[i])
        obs[14] = float(cp) / self.N_PATCHES if cp >= 0 else -0.1
        obs[15] = float(n_near) / 5.0
        for ki, (ddx, ddy) in enumerate([(-1,0),(1,0),(0,-1),(0,1)]):
            obs[16 + ki] = self.food[(x+ddx)%g, (y+ddy)%g]
        return obs

    def step(self, actions):
        self.prev_pos = self.pos.copy(); g = self.G
        rewards = np.full(C.N_AGENTS, -C.MOVE_COST)
        dx_map = {0:-1,1:1,2:0,3:0,4:0}; dy_map = {0:0,1:0,2:-1,3:1,4:0}
        for i, a in enumerate(actions):
            if a < 4:
                self.pos[i,0] = (self.pos[i,0] + dx_map[a]) % g
                self.pos[i,1] = (self.pos[i,1] + dy_map[a]) % g
            else:
                x, y = self.pos[i]
                if self.food[x, y] > 0:
                    cp = self._cell_patch(x, y)
                    same_p = sum(1 for j in range(C.N_AGENTS)
                                 if j != i and self._cell_patch(*self.pos[j]) == cp and cp >= 0)
                    rewards[i] += max(float(self.food[x, y]) - same_p * 0.08, 0.1)
                    self.food[x, y] = 0.0
        for p, cells in enumerate(self.patch_cells):
            rv = self.PATCH_REWARDS[p]
            for cx, cy in cells:
                if self.food[cx,cy] == 0.0 and np.random.rand() < self.RESPAWN_RATE:
                    self.food[cx, cy] = rv
        self.reward_ema = self.reward_ema * 0.95 + rewards * 0.05
        return rewards

    def get_all_obs(self): return [self._obs(i) for i in range(C.N_AGENTS)]


# ---------------------------------------------------------------------------
# POLICY GRADIENT UPDATE
# ---------------------------------------------------------------------------
def policy_gradient_update(agent):
    if not TORCH or not agent.ep_log_probs: return 0.0
    returns = []
    G = 0.0
    for r in reversed(agent.ep_rewards):
        G = r + C.GAMMA_RL * G; returns.insert(0, G)
    returns_t = torch.FloatTensor(returns)
    returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
    lp  = torch.stack(agent.ep_log_probs) if isinstance(agent.ep_log_probs[0], torch.Tensor) \
          else torch.FloatTensor(agent.ep_log_probs)
    ent = torch.FloatTensor(agent.ep_entropies)
    alg = torch.FloatTensor(agent.ep_tau_aligns)
    wt  = (1.0 + alg * C.TAU_ALIGN_W).clamp(0.7, 1.5)
    loss = -(lp * returns_t * wt).mean() - ent.mean() * C.ENTROPY_COEF
    agent.optimizer.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), 0.5)
    agent.optimizer.step()
    return float(loss.item())


# ---------------------------------------------------------------------------
# TRAINING LOOP
# ---------------------------------------------------------------------------
def train():
    print("=" * 70)
    print("SYMPHONON - Variational Multi-Agent v16 (REINFORCE baseline + Psi-gated tau coupling)")
    print(f"  Agents: {C.N_AGENTS}  |  tau-dim: {C.TAU_DIM}  |  Episodes: {C.N_EPISODES}")
    print(f"  Backend: {'PyTorch' if TORCH else 'NumPy'}")
    print("=" * 70)

    Agent._uid_counter = 0
    agents  = [Agent() for _ in range(C.N_AGENTS)]
    env     = ForagingGrid()
    rd      = RegimeDetector()

    # Reset phase state
    VariationalField._anchors       = {}
    VariationalField._low_rri_count = 0
    VariationalField._recrystal_on  = False
    VariationalField._recrystal_cnt = 0

    lyap_win = deque(maxlen=C.LYAP_WINDOW)
    f_prev   = 0.0
    history  = {'F':[], 'psi':[], 'n_regimes':[], 'gfc':[],
                 'reward':[], 'lyap':[], 'rri':[], 'df_dt':[]}
    t0 = time.time()

    for episode in range(C.N_EPISODES):
        env.reset()
        for a in agents: a.reset_episode()
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
                agent.ep_log_probs.append(torch.tensor(log_ps[i]) if TORCH else log_ps[i])
                agent.ep_rewards.append(float(rewards[i]))
                agent.ep_entropies.append(ents[i])
                tn_i  = agent.tau / (np.linalg.norm(agent.tau) + 1e-8)
                agent.ep_tau_aligns.append(float(np.dot(tn_i, tau_mean_n)))

            # Social update every step, patch-aware competition
            for i, a in enumerate(agents):
                for j, b in enumerate(agents):
                    if i >= j: continue
                    pi = env._cell_patch(*env.pos[i]); pj = env._cell_patch(*env.pos[j])
                    compete = 1.0 if (pi == pj and pi >= 0) else 0.1
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

        f_val   = VariationalField.compute_F(psi, rri, gfc, psi_p, reward_norm)
        df_dt   = f_val - f_prev; f_prev = f_val
        lyap_win.append(1 if df_dt < 0 else 0)
        lyap_pct = float(np.mean(list(lyap_win))) if lyap_win else 0.5

        VariationalField.langevin_step(agents, psi, rri, gfc, psi_p, reward_norm)

        # v15: inline REINFORCE + tau bias update (works without PyTorch)
        mean_r_ep = float(np.mean(ep_rewards_all))
        for agent in agents:
            # REINFORCE with baseline + Psi-gated tau bias
            agent_r = float(np.mean(agent.ep_rewards)) if agent.ep_rewards else mean_r_ep
            agent.policy.update(agent_r, mean_r_ep, agent.tau, psi)

        history['F'].append(f_val); history['psi'].append(psi)
        history['n_regimes'].append(n_reg); history['gfc'].append(gfc)
        history['reward'].append(mean_r); history['lyap'].append(lyap_pct)
        history['rri'].append(rri); history['df_dt'].append(df_dt)

        if (episode + 1) % C.LOG_EVERY == 0:
            recrystal_flag = " [RECRYST]" if VariationalField._recrystal_on else ""
            n_anch = len(VariationalField._anchors)
            anchor_flag    = f" [A:{n_anch}]" if n_anch > 0 else ""
            psi_icon = "!" if psi > 0.70 else ("~" if psi > 0.50 else "OK")
            lyap_icon = "v" if df_dt < 0 else "^"
            print(
                f"ep {episode+1:>4d} | "
                f"F={f_val:>+7.4f} dF={df_dt:>+7.4f}{lyap_icon} | "
                f"Psi={psi:.3f}{psi_icon} reg={n_reg} | "
                f"RRI={rri:.3f} GFC={gfc:.3f} | "
                f"LYAP={lyap_pct*100:.0f}% R={mean_r:>+6.3f} | "
                f"t={time.time()-t0:.0f}s{recrystal_flag}{anchor_flag}"
            )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    _summary(history)
    return agents, history


def _summary(h):
    n = len(h['F']); last = max(1, n // 5)
    print(f"\n  Final {last} episodes:")
    print(f"  F mean:      {np.mean(h['F'][-last:]):>+8.4f}")
    print(f"  Psi mean:    {np.mean(h['psi'][-last:]):>8.3f}")
    print(f"  RRI mean:    {np.mean(h['rri'][-last:]):>8.3f}")
    print(f"  N reg mean:  {np.mean(h['n_regimes'][-last:]):>8.1f}")
    print(f"  GFC mean:    {np.mean(h['gfc'][-last:]):>8.3f}")
    print(f"  LYAP:        {np.mean(h['lyap'][-last:])*100:>7.1f}%")
    print(f"  Reward mean: {np.mean(h['reward'][-last:]):>+8.3f}")
    lyap_g = float(np.mean(h['lyap']))
    rri_g  = float(np.mean(h['rri']))
    n_g    = float(np.mean(h['n_regimes'][-last:]))
    print()
    print(f"  Lyapunov global ({lyap_g*100:.1f}%): {'VERIFIED OK' if lyap_g > 0.55 else 'PARTIAL ~'}")
    print(f"  RRI global ({rri_g:.3f}): {'STABLE REGIMES OK' if rri_g > 0.10 else 'FLUID ~'}")
    if   n_g >= 3: print(f"  Regime pluralism ({n_g:.1f}): STRUCTURED PLURALISM OK")
    elif n_g >= 2: print(f"  Regime pluralism ({n_g:.1f}): COEXISTENCE ~")
    else:          print(f"  Regime pluralism ({n_g:.1f}): MONO-REGIME !")


def analyze(agents):
    rd = RegimeDetector()
    rl, sizes, centroids, n_reg, dom_idx, dom_frac = rd.detect(agents)
    print("\n" + "=" * 70)
    print("  AGENT ANALYSIS POST-TRAINING")
    print("=" * 70)
    print(f"  Distinct regimes: {n_reg}  |  Dominant: {dom_idx}  ({dom_frac*100:.1f}%)")
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    for i, a in enumerate(agents):
        lbl = int(rl[i]) if i < len(rl) else -1
        ch  = chars[lbl % len(chars)] if lbl >= 0 else "?"
        tn  = float(np.linalg.norm(a.tau))
        bar = chr(9608) * int(tn * 10)
        print(f"  Agent {a.uid:>2d} [R:{ch}] tau={tn:.3f} {bar:<10s} "
              f"curv={a.tau_curv:.4f} ali={a._ali_score:.3f} ant={a._ant_score:.3f}")
    if len(centroids) >= 2:
        print("\n  Regime separation (cosine):")
        for i in range(len(centroids)):
            for j in range(i+1, len(centroids)):
                cs = float(np.dot(centroids[i], centroids[j]) /
                           (np.linalg.norm(centroids[i]) * np.linalg.norm(centroids[j]) + 1e-8))
                print(f"    R{i}-R{j}: cos={cs:.3f}  sep={1-cs:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    np.random.seed(42)
    if TORCH: torch.manual_seed(42)
    agents, history = train()
    analyze(agents)
