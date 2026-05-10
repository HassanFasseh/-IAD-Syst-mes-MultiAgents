"""
train.py — Entraînement de l'agent Q-Learning
Usage : python train.py [--scenario equilibre|asymetrique] [--episodes 500]
"""

import argparse
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from intersection import Intersection, SCENARIOS
from agent import QLearningAgent, FixedCycleBaseline


def run_episode(env, agent, n_steps=200, train=True):
    """Exécute un épisode complet. Retourne la récompense totale."""
    state = env.reset()
    state_idx = env.state_to_index(state)
    total_reward = 0

    for _ in range(n_steps):
        if train:
            action = agent.choose_action(state_idx)
        else:
            action = agent.greedy_action(state_idx)

        next_state, reward, _ = env.step(action)
        next_idx = env.state_to_index(next_state)

        if train:
            agent.update(state_idx, action, reward, next_idx)

        state_idx = next_idx
        total_reward += reward

    return total_reward


def run_baseline_episode(env, baseline, n_steps=200):
    """Exécute un épisode avec la politique baseline."""
    env.reset()
    baseline.reset()
    total_reward = 0
    for _ in range(n_steps):
        action = baseline.choose_action(0)
        _, reward, _ = env.step(action)
        total_reward += reward
    return total_reward


def train(
    scenario='equilibre',
    n_episodes=500,
    n_steps=200,
    alpha=0.1,
    gamma=0.95,
    epsilon_decay=0.995,
    seed=42,
    callback=None,
):
    """
    Lance l'entraînement complet.
    callback(ep, reward, epsilon) est appelé à chaque épisode si fourni.
    Retourne les historiques de récompenses (QL et baseline).
    """
    cfg = SCENARIOS[scenario]
    env = Intersection(lambdas=cfg['lambdas'], seed=seed)
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=alpha,
        gamma=gamma,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=epsilon_decay,
        seed=seed,
    )
    baseline = FixedCycleBaseline(period=5)

    ql_rewards = []
    baseline_rewards = []

    for ep in range(n_episodes):
        # Q-Learning
        r_ql = run_episode(env, agent, n_steps=n_steps, train=True)
        agent.decay_epsilon()
        ql_rewards.append(r_ql)

        # Baseline (même environnement, même seed pour comparaison équitable)
        env_b = Intersection(lambdas=cfg['lambdas'], seed=seed + ep)
        r_base = run_baseline_episode(env_b, baseline, n_steps=n_steps)
        baseline_rewards.append(r_base)

        if callback:
            callback(ep, r_ql, agent.epsilon)

    return agent, ql_rewards, baseline_rewards, env


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', default='equilibre', choices=list(SCENARIOS.keys()))
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.95)
    args = parser.parse_args()

    print(f"Entraînement — Scénario : {args.scenario} | Épisodes : {args.episodes}")

    def cb(ep, r, eps):
        if ep % 50 == 0:
            print(f"  Épisode {ep:4d} | Récompense : {r:8.1f} | ε = {eps:.3f}")

    agent, ql_r, base_r, env = train(
        scenario=args.scenario,
        n_episodes=args.episodes,
        n_steps=args.steps,
        alpha=args.alpha,
        gamma=args.gamma,
        callback=cb,
    )

    print(f"\nRécompense moy. Q-Learning (100 derniers) : {np.mean(ql_r[-100:]):.1f}")
    print(f"Récompense moy. Baseline   (100 derniers) : {np.mean(base_r[-100:]):.1f}")

    # Sauvegarde table Q
    os.makedirs('outputs', exist_ok=True)
    np.save(f'outputs/Q_table_{args.scenario}.npy', agent.Q)
    print(f"Table Q sauvegardée dans outputs/Q_table_{args.scenario}.npy")
