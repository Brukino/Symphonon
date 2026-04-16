"""
SYMPHONON Ω v4.6.1 — The Emergent Organism
===========================================
FUSIONE FINALE: Estetica Bio-Fluida + Logica Evolutiva Spaziale.
Feedback Causale Reale (Locale/Globale) + Movimento guidato dal Fallimento.

Fix e miglioramenti rispetto a v4.6:
─────────────────────────────────────
  BUG 1  pygame.draw.line con array NumPy float → crash su pygame 2.x
          Ora le coordinate sono convertite a tuple int prima del draw.
  BUG 2  Dead code nei colori Identity: il ramo else (grigio) era
          irraggiungibile — drift_factor > 0.1 è sempre True quando
          success <= 0.1 (poiché drift_factor = max(0, 0.35 − success)).
          Logica colori corretta: verde / arancio / grigio su soglie distinte.
  BUG 3  Loop principale a livello di modulo: nessun main(), impossibile
          importare senza avviare pygame. Ora incapsulato in main().

Aggiunte:
─────────
  · HUD con metriche in tempo reale (stabilità globale, n. identità, FPS)
  · SPACE = pausa / riprendi
  · R     = reset completo (campo + particelle + identità)
  · ESC   = esci
  · LMB   = spawna un'identità nella posizione cliccata
  · RMB   = perturba il campo nella posizione cliccata
"""

import pygame
import numpy as np
import sys
import os

os.environ['SDL_VIDEO_CENTERED'] = '1'

# ═══════════════════════════════════════
# CONFIGURAZIONE
# ═══════════════════════════════════════
WIN_W, WIN_H     = 1024, 720
GRID             = 70
NUM_PARTICLES    = 3600
FPS              = 60
TRAIL_ALPHA      = 25

DT, DIFF, NONLIN, DAMP = 0.15, 0.18, 0.05, 0.992
MAX_IDENTITIES   = 40
L_G_RATIO        = 0.65   # 65% Locale, 35% Globale

# Colori HUD
C_HUD   = (180, 210, 255)
C_DIM   = (80,  100, 140)
C_GRN   = (80,  240, 160)
C_AMB   = (255, 180,  50)
C_RED   = (255,  70,  60)


# ═══════════════════════════════════════
# MOTORE DEL CAMPO (Fisica)
# ═══════════════════════════════════════
class Field:
    def __init__(self):
        self.A = (np.random.randn(GRID, GRID) +
                  1j * np.random.randn(GRID, GRID)) * 0.4

    def step(self):
        lap = (np.roll(self.A,  1, 0) + np.roll(self.A, -1, 0) +
               np.roll(self.A,  1, 1) + np.roll(self.A, -1, 1) - 4 * self.A)
        self.A += DT * (DIFF * lap - NONLIN * (np.abs(self.A) ** 2) * self.A)
        self.A *= DAMP
        amp = np.abs(self.A)
        mask = amp > 2.0
        self.A[mask] *= 2.0 / amp[mask]
        self.A = np.nan_to_num(self.A)

    def get_metrics(self):
        phase = np.angle(self.A)
        gy_arr, gx_arr = np.gradient(phase)          # numpy: [d_row, d_col]
        om = 1.0 - np.tanh(np.sqrt(gx_arr ** 2 + gy_arr ** 2))
        return phase, om, float(np.mean(om))

    def perturb(self, gx, gy, radius=4, strength=1.2):
        """Perturbazione locale del campo (click RMB)."""
        x0 = max(0, gx - radius); x1 = min(GRID, gx + radius + 1)
        y0 = max(0, gy - radius); y1 = min(GRID, gy + radius + 1)
        noise = np.random.uniform(0, 2 * np.pi, (x1 - x0, y1 - y0))
        self.A[x0:x1, y0:y1] *= strength * np.exp(1j * noise).astype(np.complex64)


# ═══════════════════════════════════════
# IDENTITÀ ADATTIVA (L'Agente)
# ═══════════════════════════════════════
class Identity:
    def __init__(self, x, y):
        self.x, self.y    = float(x), float(y)
        self.vx, self.vy  = 0.0, 0.0
        self.success      = 0.1
        self.prev_local_om = 0.0
        self.pulse        = np.random.rand() * 6.28
        self.color        = (200, 200, 200)

    def act(self, field_obj, omega_field):
        gx = int(self.x) % GRID
        gy = int(self.y) % GRID
        self.prev_local_om = omega_field[gx, gy]
        influence = (np.random.randn() * 0.12 + self.success * 1.2) * self.prev_local_om
        field_obj.A[gx, gy] *= np.exp(1j * influence)
        return abs(influence) * 0.025

    def update(self, new_omega_field, global_delta, cost, phase_field):
        gx = int(self.x) % GRID
        gy = int(self.y) % GRID
        current_om = new_omega_field[gx, gy]

        # 1. FEEDBACK CAUSALE: Ho migliorato la situazione?
        delta_local = current_om - self.prev_local_om

        # 2. REWARD INTEGRATO
        reward = (L_G_RATIO * delta_local) + ((1 - L_G_RATIO) * global_delta)
        self.success = self.success * 0.9 + (reward - cost)

        # 3. MOVIMENTO ADATTIVO
        # Se il successo è basso l'agente "perde aderenza" e scivola lungo
        # il gradiente di fase (comportamento nomade).
        drift_factor = max(0.0, 0.35 - self.success)
        angle = phase_field[gx, gy]

        self.vx = self.vx * 0.8 + np.cos(angle) * drift_factor
        self.vy = self.vy * 0.8 + np.sin(angle) * drift_factor

        self.x = (self.x + self.vx) % GRID
        self.y = (self.y + self.vy) % GRID

        # 4. STATO VISIVO  ← FIX: logica colori corretta (3 stati distinti)
        self.pulse += 0.1 + max(0, self.success) * 0.2
        if self.success > 0.1:
            self.color = (160, 255, 200)   # Verde/Ciano — Stanziale
        elif self.success > -0.05:
            self.color = (255, 180, 80)    # Arancio     — Nomade
        else:
            self.color = (150, 150, 150)   # Grigio      — In declino


# ═══════════════════════════════════════
# MARE PARTICELLARE (Visualizer)
# ═══════════════════════════════════════
class ParticleSea:
    def __init__(self, n):
        self.pos       = np.random.rand(n, 2) * [WIN_W, WIN_H]
        self.vel       = np.zeros((n, 2))
        self.max_speed = 3.8

    def update(self, phase, omega):
        gx = (self.pos[:, 0] * GRID // WIN_W).astype(int) % GRID
        gy = (self.pos[:, 1] * GRID // WIN_H).astype(int) % GRID

        target_vx = np.cos(phase[gx, gy]) * self.max_speed
        target_vy = np.sin(phase[gx, gy]) * self.max_speed

        steer = 0.1 + (1.0 - omega[gx, gy]) * 0.2
        self.vel[:, 0] += (target_vx - self.vel[:, 0]) * steer
        self.vel[:, 1] += (target_vy - self.vel[:, 1]) * steer

        self.pos += self.vel
        self.pos[:, 0] %= WIN_W
        self.pos[:, 1] %= WIN_H

    def draw(self, surface, omega):
        gx = (self.pos[:, 0] * GRID // WIN_W).astype(int) % GRID
        gy = (self.pos[:, 1] * GRID // WIN_H).astype(int) % GRID
        oms = omega[gx, gy]

        # FIX: converte coordinate NumPy float → tuple int per pygame 2.x
        for i in range(0, len(self.pos), 2):
            lum   = int(oms[i] * 180 + 75)
            color = (lum // 5, lum // 2, lum)
            p1 = (int(self.pos[i, 0]),
                  int(self.pos[i, 1]))
            p2 = (int(self.pos[i, 0] - self.vel[i, 0] * 1.2),
                  int(self.pos[i, 1] - self.vel[i, 1] * 1.2))
            pygame.draw.line(surface, color, p1, p2, 1)


# ═══════════════════════════════════════
# HUD
# ═══════════════════════════════════════
def draw_hud(screen, font_sm, font_xs, tick, fps, stability, n_ids, paused):
    lines = [
        (f"SYMPHONON Ω v4.6.1",        C_HUD),
        (f"tick       {tick:>8d}",      C_DIM),
        (f"fps        {fps:>8.1f}",     C_DIM),
        (f"stabilità  {stability:>8.4f}", C_GRN if stability > 0.5 else C_AMB),
        (f"identità   {n_ids:>8d}",     C_HUD),
    ]
    x, y = 10, 10
    for txt, col in lines:
        surf = font_sm.render(txt, True, col)
        screen.blit(surf, (x, y))
        y += 16

    if paused:
        ps = font_sm.render("— PAUSA —", True, C_AMB)
        screen.blit(ps, (WIN_W // 2 - ps.get_width() // 2, 10))

    # Barra di stabilità
    bar_w = 130; bar_h = 6
    bx, by = 10, y + 4
    pygame.draw.rect(screen, (20, 30, 55), (bx, by, bar_w, bar_h), border_radius=3)
    fill = int(np.clip(stability, 0, 1) * bar_w)
    if fill > 0:
        col_bar = C_GRN if stability > 0.5 else C_AMB
        pygame.draw.rect(screen, col_bar, (bx, by, fill, bar_h), border_radius=3)
    pygame.draw.rect(screen, C_DIM, (bx, by, bar_w, bar_h), 1, border_radius=3)

    # Tasti
    helps = [
        "LMB spawna · RMB perturba",
        "SPACE pausa · R reset · ESC esci",
    ]
    y2 = WIN_H - 8
    for h in reversed(helps):
        hs = font_xs.render(h, True, C_DIM)
        screen.blit(hs, (10, y2 - hs.get_height()))
        y2 -= hs.get_height() + 2


# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════
def make_objects():
    """Crea/resetta tutti gli oggetti di simulazione."""
    return Field(), ParticleSea(NUM_PARTICLES), []


def main():
    pygame.init()
    screen  = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("SYMPHONON Ω v4.6.1 — Emergent Organism")
    clock   = pygame.time.Clock()

    font_sm = pygame.font.SysFont("monospace", 13, bold=False)
    font_xs = pygame.font.SysFont("monospace", 11, bold=False)

    field, sea, identities = make_objects()
    overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)

    global_prev_stability = 0.5
    paused = False
    tick   = 0
    fps    = 60.0

    running = True
    while running:
        # ── Metriche pre-azione ────────────────────────────────────────
        phase, omega, global_stability = field.get_metrics()
        delta_global = global_stability - global_prev_stability
        global_prev_stability = global_stability

        # ── Eventi ────────────────────────────────────────────────────
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.key == pygame.K_r:
                    field, sea, identities = make_objects()
                    global_prev_stability = 0.5
                    tick = 0

            elif e.type == pygame.MOUSEBUTTONDOWN:
                mx, my = e.pos
                gx_m = int(mx * GRID / WIN_W) % GRID
                gy_m = int(my * GRID / WIN_H) % GRID
                if e.button == 1:   # LMB → spawna identità
                    if len(identities) < MAX_IDENTITIES:
                        identities.append(Identity(gx_m, gy_m))
                elif e.button == 3: # RMB → perturba campo
                    field.perturb(gx_m, gy_m)

        if paused:
            clock.tick(FPS)
            # Ridisegna solo per mostrare il cartello PAUSA
            draw_hud(screen, font_sm, font_xs, tick, fps,
                     global_stability, len(identities), paused)
            pygame.display.flip()
            continue

        # ── Logica: Azione → Fisico → Feedback ────────────────────────
        costs = {}
        for iden in identities:
            costs[iden] = iden.act(field, omega)

        field.step()
        new_phase, new_omega, _ = field.get_metrics()

        for iden in identities:
            iden.update(new_omega, delta_global, costs[iden], new_phase)

        # Selezione naturale
        identities = [i for i in identities if i.success > -0.2]

        # Spawn automatico in zone stabili
        if len(identities) < MAX_IDENTITIES and np.random.rand() > 0.85:
            coords = np.argwhere(new_omega > 0.8)
            if len(coords) > 0:
                c = coords[np.random.randint(len(coords))]
                identities.append(Identity(c[0], c[1]))

        # ── Rendering ─────────────────────────────────────────────────
        overlay.fill((5, 10, 20, TRAIL_ALPHA))
        screen.blit(overlay, (0, 0))

        sea.update(new_phase, new_omega)
        sea.draw(screen, new_omega)

        for iden in identities:
            px = int(iden.x * WIN_W / GRID)
            py = int(iden.y * WIN_H / GRID)
            sz = max(2, int(5 + np.sin(iden.pulse) * 3))
            pygame.draw.circle(screen, iden.color, (px, py), sz, 1)
            if iden.success > 0.1:   # aura di stabilità
                pygame.draw.circle(screen, iden.color, (px, py), sz + 8, 1)

        draw_hud(screen, font_sm, font_xs, tick, fps,
                 global_stability, len(identities), paused)

        pygame.display.flip()
        dt_ms = clock.tick(FPS)
        fps   = 1000.0 / max(dt_ms, 1)
        tick += 1

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
