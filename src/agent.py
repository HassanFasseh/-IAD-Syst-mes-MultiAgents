"""
agent.py — Agent Q-Learning tabulé
Implémentation from scratch avec politique ε-greedy et decay.
"""

import numpy as np


class QLearningAgent:
    """
    Agent Q-Learning tabulé pour le contrôle d'un feu de signalisation.

    Paramètres :
        n_states  : taille de l'espace d'états
        n_actions : nombre d'actions (2 : Maintenir / Changer)
        alpha     : taux d'apprentissage (0 < α ≤ 1)
        gamma     : facteur d'actualisation (0 < γ ≤ 1)
        epsilon   : probabilité d'exploration initiale
        epsilon_min : valeur minimale de ε
        epsilon_decay : multiplicateur de decay appliqué à chaque épisode
    """

    def __init__(
        self,
        n_states,
        n_actions,
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        seed=None,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)

        # Table Q initialisée à zéro
        self.Q = np.zeros((n_states, n_actions))

    def choose_action(self, state_idx):
        """Politique ε-greedy."""
        if self.rng.random() < self.epsilon:
            return self.rng.integers(0, self.n_actions)  # Exploration
        return int(np.argmax(self.Q[state_idx]))  # Exploitation

    def update(self, state_idx, action, reward, next_state_idx):
        """Mise à jour de Bellman."""
        best_next = np.max(self.Q[next_state_idx])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.Q[state_idx, action]
        self.Q[state_idx, action] += self.alpha * td_error

    def decay_epsilon(self):
        """Applique le decay de ε à la fin d'un épisode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def greedy_action(self, state_idx):
        """Action greedy pure (sans exploration), pour l'évaluation."""
        return int(np.argmax(self.Q[state_idx]))

    def reset_exploration(self, epsilon=1.0):
        self.epsilon = epsilon


class FixedCycleBaseline:
    """
    Politique de référence : feu à alternance périodique fixe.
    Change de phase toutes les `period` étapes.
    """

    def __init__(self, period=5):
        self.period = period
        self.t = 0

    def choose_action(self, state_idx):
        action = 1 if (self.t % self.period == 0) else 0
        self.t += 1
        return action

    def reset(self):
        self.t = 0
