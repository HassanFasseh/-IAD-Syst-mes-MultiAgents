"""
intersection.py — Simulation du carrefour à 4 branches
Module séparable de l'agent.
"""

import numpy as np

# Paramètres par défaut
MAX_QUEUE = 5       # Niveaux de file : 0..5 véhicules (discrétisé)
LAMBDA_DEFAULT = [0.3, 0.3, 0.3, 0.3]  # Taux d'arrivée Poisson par branche N,S,E,O
FLOW_RATE = 2       # Véhicules qui passent par pas de temps si phase active


class Intersection:
    """
    Carrefour à 4 branches (N, S, E, O).

    Phases :
        0 = N/S vert  (Nord et Sud passent)
        1 = E/O vert  (Est et Ouest passent)
        2 = Transition orange (personne ne passe, dure 1 pas)

    Actions :
        0 = Maintenir la phase courante
        1 = Demander un changement de phase

    Logique de transition :
        - Si on est en phase active (0 ou 1) et l'agent choisit Changer
          → on passe en TRANSITION pour 1 pas
        - Au pas suivant, la transition se résout automatiquement
          vers la phase opposée (l'action est ignorée pendant TRANSITION)
        - Si on est en phase active et l'agent choisit Maintenir → on reste
    """

    PHASE_NS    = 0
    PHASE_EO    = 1
    PHASE_TRANS = 2

    def __init__(self, lambdas=None, max_queue=MAX_QUEUE, flow_rate=FLOW_RATE, seed=None):
        self.lambdas   = lambdas if lambdas is not None else LAMBDA_DEFAULT[:]
        self.max_queue = max_queue
        self.flow_rate = flow_rate
        self.rng       = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        """Remet le carrefour à zéro."""
        self.queues      = np.zeros(4, dtype=int)
        self.phase       = self.PHASE_NS
        self.prev_phase  = self.PHASE_NS   # phase active avant la transition
        self.total_waiting = 0
        self.step_count  = 0
        return self._get_state()

    def _get_state(self):
        return tuple(self.queues.tolist()) + (self.phase,)

    def step(self, action):
        """
        Exécute une action et retourne (état_suivant, récompense, info).
        """
        # --- 1. Arrivées stochastiques (Poisson) ---
        arrivals = self.rng.poisson(self.lambdas)
        for i in range(4):
            self.queues[i] = min(self.queues[i] + arrivals[i], self.max_queue)

        # --- 2. Transition de phase ---
        if self.phase == self.PHASE_TRANS:
            # Transition se résout vers la phase opposée (action ignorée)
            self.phase = self.PHASE_EO if self.prev_phase == self.PHASE_NS else self.PHASE_NS
        else:
            # Phase active : l'agent décide
            if action == 1:
                self.prev_phase = self.phase
                self.phase = self.PHASE_TRANS
            # action == 0 : Maintenir → rien ne change

        # --- 3. Écoulement du trafic ---
        if self.phase == self.PHASE_NS:
            self.queues[0] = max(0, self.queues[0] - self.flow_rate)
            self.queues[1] = max(0, self.queues[1] - self.flow_rate)
        elif self.phase == self.PHASE_EO:
            self.queues[2] = max(0, self.queues[2] - self.flow_rate)
            self.queues[3] = max(0, self.queues[3] - self.flow_rate)
        # PHASE_TRANS : personne ne passe

        # --- 4. Récompense ---
        waiting = int(np.sum(self.queues))
        reward  = -waiting
        self.total_waiting += waiting
        self.step_count    += 1

        info = {
            'arrivals': arrivals.tolist(),
            'queues':   self.queues.copy(),
            'phase':    self.phase,
            'waiting':  waiting,
        }
        return self._get_state(), reward, info

    def state_to_index(self, state):
        """Encode un état tuple en entier (index dans la table Q)."""
        q0, q1, q2, q3, phase = state
        idx = q0
        idx = idx * (self.max_queue + 1) + q1
        idx = idx * (self.max_queue + 1) + q2
        idx = idx * (self.max_queue + 1) + q3
        idx = idx * 3 + phase
        return idx

    def get_state_space_size(self):
        return (self.max_queue + 1) ** 4 * 3

    @property
    def n_actions(self):
        return 2

    @property
    def n_states(self):
        return self.get_state_space_size()


# ── Scénarios de trafic ───────────────────────────────────────────
SCENARIOS = {
    'equilibre': {
        'lambdas': [0.3, 0.3, 0.3, 0.3],
        'label':   'Trafic équilibré'
    },
    'asymetrique': {
        'lambdas': [0.7, 0.7, 0.1, 0.1],
        'label':   'Charge asymétrique (N/S forte)'
    },
}
