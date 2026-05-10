"""
evaluate.py — Évaluation et génération des figures de convergence
Usage : python evaluate.py [--scenario equilibre|asymetrique]
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(__file__))
from intersection import Intersection, SCENARIOS
from agent import QLearningAgent, FixedCycleBaseline
from train import train, run_episode, run_baseline_episode


def smooth(data, window=20):
    """Moyenne glissante pour lisser les courbes."""
    return np.convolve(data, np.ones(window) / window, mode='valid')


def plot_convergence(ql_rewards, baseline_rewards, scenario_label, save_path=None):
    """Courbes de récompense cumulée et comparaison avec la baseline."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'Convergence Q-Learning — {scenario_label}', fontsize=14, fontweight='bold')

    # --- Graphe 1 : Récompenses brutes + lissées ---
    ax = axes[0]
    eps = range(len(ql_rewards))
    ax.plot(eps, ql_rewards, alpha=0.3, color='steelblue', linewidth=0.8, label='Q-Learning (brut)')
    ax.plot(range(len(smooth(ql_rewards))), smooth(ql_rewards), color='steelblue', linewidth=2, label='Q-Learning (lissé)')
    ax.plot(eps, baseline_rewards, alpha=0.3, color='tomato', linewidth=0.8, label='Baseline (brut)')
    ax.plot(range(len(smooth(baseline_rewards))), smooth(baseline_rewards), color='tomato', linewidth=2, label='Baseline (lissé)')
    ax.set_xlabel('Épisode')
    ax.set_ylabel('Récompense totale')
    ax.set_title('Évolution de la récompense')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Graphe 2 : Comparaison finale (boxplot 100 derniers épisodes) ---
    ax2 = axes[1]
    last_ql = ql_rewards[-100:]
    last_base = baseline_rewards[-100:]
    bp = ax2.boxplot([last_ql, last_base], tick_labels=['Q-Learning', 'Baseline fixe'], patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][1].set_facecolor('tomato')
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color='black')
    ax2.set_title('Distribution (100 derniers épisodes)')
    ax2.set_ylabel('Récompense totale')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure sauvegardée : {save_path}")
    plt.close()
    return fig


def plot_q_heatmap(agent, env, phase=0, save_path=None):
    """
    Visualise la table Q pour une tranche de l'espace d'états.
    On fixe q_E=0, q_O=0 et on varie q_N et q_S (phase fixée).
    """
    max_q = env.max_queue
    Z_maintain = np.zeros((max_q + 1, max_q + 1))
    Z_change = np.zeros((max_q + 1, max_q + 1))
    Z_best = np.zeros((max_q + 1, max_q + 1))

    for qN in range(max_q + 1):
        for qS in range(max_q + 1):
            state = (qN, qS, 0, 0, phase)
            idx = env.state_to_index(state)
            Z_maintain[qN, qS] = agent.Q[idx, 0]
            Z_change[qN, qS] = agent.Q[idx, 1]
            Z_best[qN, qS] = np.argmax(agent.Q[idx])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    phase_name = ['N/S vert', 'E/O vert', 'Transition'][phase]
    fig.suptitle(f'Table Q (phase = {phase_name}, q_E=0, q_O=0)', fontsize=13, fontweight='bold')

    for ax, Z, title, cmap in zip(
        axes,
        [Z_maintain, Z_change, Z_best],
        ['Q(s, Maintenir)', 'Q(s, Changer)', 'Action optimale (0=Maintenir, 1=Changer)'],
        ['Blues', 'Reds', 'RdYlGn'],
    ):
        im = ax.imshow(Z, origin='lower', cmap=cmap, aspect='auto')
        ax.set_xlabel('q_S')
        ax.set_ylabel('q_N')
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure sauvegardée : {save_path}")
    plt.close()
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', default='equilibre', choices=list(SCENARIOS.keys()))
    parser.add_argument('--episodes', type=int, default=500)
    args = parser.parse_args()

    os.makedirs('figures', exist_ok=True)
    cfg = SCENARIOS[args.scenario]

    print(f"Entraînement pour évaluation — Scénario : {cfg['label']}")
    agent, ql_r, base_r, env = train(scenario=args.scenario, n_episodes=args.episodes)

    plot_convergence(ql_r, base_r, cfg['label'],
                     save_path=f'figures/convergence_{args.scenario}.png')
    plot_q_heatmap(agent, env, phase=0,
                   save_path=f'figures/q_heatmap_{args.scenario}.png')

    print("\nÉvaluation terminée. Figures exportées dans figures/")
