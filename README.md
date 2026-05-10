Gestion Intelligente du Trafic Urbain
Projet| Cycle Ingénieur 2025–2026

Agent Q-Learning tabulé pour l'optimisation d'un carrefour à 4 branches.


Installation

bash
pip install numpy matplotlib


> Python 3.8+ requis. Tkinter est inclus avec Python par défaut.



Lancement

#Interface graphique
bash
cd src
python gui.py


#Ligne de commande — Entraînement
bash
cd src
python train.py --scenario equilibre --episodes 500
python train.py --scenario asymetrique --episodes 500


#Ligne de commande — Évaluation + export figures
bash
cd src
python evaluate.py --scenario equilibre --episodes 500




Utilisation de l'interface

1. **Choisir un scénario** : trafic équilibré ou asymétrique (N/S surchargé)
2. **Régler les paramètres** : épisodes, α, γ, ε decay
3. **Cliquer sur "Entraîner"** : la convergence s'affiche en temps réel
4. **Simuler l'agent** : cliquer sur "Simuler l'agent" pour voir le carrefour en direct
5. **Comparer** : switcher entre mode QL et Baseline dans la simulation



Paramètres par défaut justifiés

| Paramètre          | Valeur | Explication simple                          |
| ------------------ | ------ | ------------------------------------------- |
| α (learning rate)  | 0.10   | Apprentissage stable et progressif          |
| γ (discount)       | 0.95   | L’agent prend en compte le futur            |
| ε initial          | 1.0    | Exploration complète au départ              |
| ε decay            | 0.995  | Réduction progressive de l’exploration      |
| ε_min              | 0.05   | Garde un peu d’exploration                  |
| Files discrétisées | 0–5    | Bon équilibre entre précision et simplicité |




Scénarios

- **Équilibré** : λ = [0.3, 0.3, 0.3, 0.3] pour N, S, E, O
- **Asymétrique** : λ = [0.7, 0.7, 0.1, 0.1] — forte charge N/S



MDP — Résumé

- **États** : (q_N, q_S, q_E, q_O, phase) | q_i ∈ {0..5}, phase ∈ {N/S, E/O, Transition}
- **Actions** : {0 = Maintenir, 1 = Changer de phase}
- **Récompense** : R = −Σ q_i (pénalité d'attente totale)
- **Taille de l'espace** : 6⁴ × 3 = 3888 états
