"""
gui.py — Interface graphique principale
Visualisation en temps réel de la simulation du carrefour + entraînement Q-Learning
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from collections import deque

sys.path.insert(0, os.path.dirname(__file__))
from intersection import Intersection, SCENARIOS
from agent import QLearningAgent, FixedCycleBaseline
from train import train, run_episode, run_baseline_episode


# ─────────────────────────── Couleurs ───────────────────────────
BG        = '#1a1a2e'
PANEL     = '#16213e'
ACCENT    = '#0f3460'
GREEN     = '#4ade80'
RED       = '#f87171'
ORANGE    = '#fb923c'
YELLOW    = '#fbbf24'
WHITE     = '#f1f5f9'
GRAY      = '#64748b'
BLUE      = '#60a5fa'
PURPLE    = '#c084fc'


class TrafficGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Gestion Intelligente du Trafic — Agent Q-Learning")
        self.root.configure(bg=BG)
        self.root.geometry("1280x800")
        self.root.resizable(True, True)

        # État
        self.agent = None
        self.env = None
        self.baseline = FixedCycleBaseline(period=5)
        self.training = False
        self.simulating = False
        self.sim_thread = None
        self.train_thread = None

        self.ql_rewards = []
        self.base_rewards = []
        self.reward_window = deque(maxlen=200)  # pour affichage live
        self.episode_count = 0
        self.step_count = 0

        self._build_ui()
        self._init_env()

    # ─────────────────────── Construction UI ───────────────────────

    def _build_ui(self):
        # Titre
        title_frame = tk.Frame(self.root, bg=BG)
        title_frame.pack(fill='x', padx=15, pady=(12, 4))
        tk.Label(title_frame, text="🚦 Gestion Intelligente du Trafic Urbain",
                 font=('Courier New', 17, 'bold'), bg=BG, fg=WHITE).pack(side='left')
        tk.Label(title_frame, text="IAD & SMA — Projet Partie 1",
                 font=('Courier New', 10), bg=BG, fg=GRAY).pack(side='right', pady=6)

        # Zone principale
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill='both', expand=True, padx=15, pady=4)

        # Colonne gauche : carrefour + contrôles
        left = tk.Frame(main, bg=BG, width=420)
        left.pack(side='left', fill='y', padx=(0, 10))
        left.pack_propagate(False)

        self._build_controls(left)
        self._build_intersection_canvas(left)
        self._build_stats(left)

        # Colonne droite : graphes
        right = tk.Frame(main, bg=BG)
        right.pack(side='left', fill='both', expand=True)
        self._build_charts(right)

    def _build_controls(self, parent):
        frame = tk.LabelFrame(parent, text=" Paramètres ", bg=PANEL, fg=WHITE,
                               font=('Courier New', 10, 'bold'), bd=1, relief='flat',
                               padx=8, pady=6)
        frame.pack(fill='x', pady=(0, 8))

        # Scénario
        row = tk.Frame(frame, bg=PANEL)
        row.pack(fill='x', pady=2)
        tk.Label(row, text="Scénario :", bg=PANEL, fg=WHITE,
                 font=('Courier New', 9), width=14, anchor='w').pack(side='left')
        self.scenario_var = tk.StringVar(value='equilibre')
        cb = ttk.Combobox(row, textvariable=self.scenario_var,
                          values=['equilibre', 'asymetrique'],
                          state='readonly', width=18, font=('Courier New', 9))
        cb.pack(side='left')
        cb.bind('<<ComboboxSelected>>', lambda e: self._init_env())

        # Épisodes
        self._make_slider(frame, "Épisodes :", 'episodes_var', 100, 1000, 400, step=50)
        # Alpha
        self._make_slider(frame, "α (apprentissage) :", 'alpha_var', 0.01, 0.5, 0.1,
                          step=0.01, fmt='{:.2f}')
        # Gamma
        self._make_slider(frame, "γ (actualisation) :", 'gamma_var', 0.8, 1.0, 0.95,
                          step=0.01, fmt='{:.2f}')
        # Epsilon decay
        self._make_slider(frame, "ε decay :", 'edecay_var', 0.980, 0.999, 0.995,
                          step=0.001, fmt='{:.3f}')

        # Boutons
        btn_frame = tk.Frame(frame, bg=PANEL)
        btn_frame.pack(fill='x', pady=(8, 2))
        self.train_btn = self._btn(btn_frame, "▶ Entraîner", self._start_training, GREEN)
        self.train_btn.pack(side='left', padx=(0, 6))
        self._btn(btn_frame, "⟳ Reset", self._reset_all, ORANGE).pack(side='left', padx=(0, 6))

        # Barre de progression
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(frame, variable=self.progress_var,
                                        maximum=100, length=360)
        self.progress.pack(fill='x', pady=(6, 2))
        self.status_var = tk.StringVar(value="Prêt. Choisissez les paramètres et lancez l'entraînement.")
        tk.Label(frame, textvariable=self.status_var, bg=PANEL, fg=YELLOW,
                 font=('Courier New', 8), wraplength=370, justify='left').pack(anchor='w')

    def _make_slider(self, parent, label, attr, from_, to, default, step=1, fmt='{:.0f}'):
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill='x', pady=2)
        tk.Label(row, text=label, bg=PANEL, fg=WHITE,
                 font=('Courier New', 9), width=20, anchor='w').pack(side='left')
        var = tk.DoubleVar(value=default)
        setattr(self, attr, var)
        val_label = tk.Label(row, text=fmt.format(default), bg=PANEL, fg=BLUE,
                             font=('Courier New', 9), width=7)
        val_label.pack(side='right')

        def on_change(v):
            val_label.config(text=fmt.format(float(v)))

        sl = ttk.Scale(row, from_=from_, to=to, variable=var,
                       orient='horizontal', length=170, command=on_change)
        sl.pack(side='left', padx=4)

    def _btn(self, parent, text, cmd, color):
        return tk.Button(parent, text=text, command=cmd,
                         bg=color, fg='#111', font=('Courier New', 9, 'bold'),
                         relief='flat', padx=10, pady=4, cursor='hand2',
                         activebackground=WHITE)

    def _build_intersection_canvas(self, parent):
        frame = tk.LabelFrame(parent, text=" Carrefour (temps réel) ", bg=PANEL, fg=WHITE,
                               font=('Courier New', 10, 'bold'), bd=1, relief='flat')
        frame.pack(fill='x', pady=(0, 8))

        self.canvas = tk.Canvas(frame, width=390, height=250, bg='#0d1117',
                                highlightthickness=0)
        self.canvas.pack(padx=6, pady=6)
        self._draw_intersection_static()

        # Boutons de simulation
        sim_frame = tk.Frame(frame, bg=PANEL)
        sim_frame.pack(pady=(0, 6))
        self.sim_btn = self._btn(sim_frame, "▶ Simuler l'agent", self._toggle_sim, BLUE)
        self.sim_btn.pack(side='left', padx=4)
        self.sim_mode = tk.StringVar(value='ql')
        ttk.Combobox(sim_frame, textvariable=self.sim_mode,
                     values=['ql', 'baseline'], state='readonly',
                     width=10, font=('Courier New', 9)).pack(side='left')

    def _build_stats(self, parent):
        frame = tk.LabelFrame(parent, text=" Statistiques ", bg=PANEL, fg=WHITE,
                               font=('Courier New', 10, 'bold'), bd=1, relief='flat',
                               padx=8, pady=6)
        frame.pack(fill='x')

        self.stat_vars = {}
        stats = [
            ('Épisode', 'ep_var', '0'),
            ('ε courant', 'eps_var', '1.000'),
            ('Récompense (moy. 20)', 'rew_var', '—'),
            ('Baseline (moy. 20)', 'base_var', '—'),
            ('Attente totale (sim)', 'wait_var', '0'),
        ]
        for label, key, default in stats:
            row = tk.Frame(frame, bg=PANEL)
            row.pack(fill='x', pady=1)
            tk.Label(row, text=label + ' :', bg=PANEL, fg=GRAY,
                     font=('Courier New', 9), width=22, anchor='w').pack(side='left')
            var = tk.StringVar(value=default)
            self.stat_vars[key] = var
            tk.Label(row, textvariable=var, bg=PANEL, fg=GREEN,
                     font=('Courier New', 9, 'bold')).pack(side='left')

    def _build_charts(self, parent):
        fig = Figure(figsize=(7.5, 7), facecolor=BG)
        fig.subplots_adjust(hspace=0.42, top=0.93)

        self.ax_reward = fig.add_subplot(211)
        self.ax_q = fig.add_subplot(212)

        for ax in [self.ax_reward, self.ax_q]:
            ax.set_facecolor(PANEL)
            for spine in ax.spines.values():
                spine.set_color(ACCENT)
            ax.tick_params(colors=GRAY, labelsize=8)
            ax.xaxis.label.set_color(GRAY)
            ax.yaxis.label.set_color(GRAY)
            ax.title.set_color(WHITE)
            ax.grid(True, color=ACCENT, alpha=0.5, linewidth=0.5)

        self.ax_reward.set_title('Récompense totale par épisode', fontsize=10)
        self.ax_reward.set_xlabel('Épisode')
        self.ax_reward.set_ylabel('Récompense')

        self.ax_q.set_title('Table Q — Action optimale (q_E=0, q_O=0, phase N/S)', fontsize=10)
        self.ax_q.set_xlabel('q_S')
        self.ax_q.set_ylabel('q_N')

        self.chart_canvas = FigureCanvasTkAgg(fig, master=parent)
        self.chart_canvas.get_tk_widget().pack(fill='both', expand=True)
        self.fig = fig

        # Initialiser le graphe Q vide
        self._update_q_heatmap_plot()

    # ─────────────────────── Intersection Canvas ───────────────────────

    def _draw_intersection_static(self):
        """Dessine le fond fixe du carrefour."""
        c = self.canvas
        W, H = 390, 250
        cx, cy = W // 2, H // 2
        road_w = 60

        # Routes
        c.create_rectangle(cx - road_w // 2, 0, cx + road_w // 2, H, fill='#1f2937', outline='')
        c.create_rectangle(0, cy - road_w // 2, W, cy + road_w // 2, fill='#1f2937', outline='')
        # Centre
        c.create_rectangle(cx - road_w // 2, cy - road_w // 2,
                            cx + road_w // 2, cy + road_w // 2, fill='#374151', outline='')
        # Marquages
        for y in range(0, cy - road_w // 2, 30):
            c.create_line(cx, y, cx, y + 15, fill='#6b7280', width=2, dash=(4, 8))
        for y in range(cy + road_w // 2, H, 30):
            c.create_line(cx, y, cx, y + 15, fill='#6b7280', width=2, dash=(4, 8))
        for x in range(0, cx - road_w // 2, 30):
            c.create_line(x, cy, x + 15, cy, fill='#6b7280', width=2, dash=(4, 8))
        for x in range(cx + road_w // 2, W, 30):
            c.create_line(x, cy, x + 15, cy, fill='#6b7280', width=2, dash=(4, 8))

        # Labels des directions
        for text, x, y in [('N', cx, 14), ('S', cx, H - 14),
                             ('E', W - 18, cy), ('O', 18, cy)]:
            c.create_text(x, y, text=text, fill=WHITE,
                          font=('Courier New', 11, 'bold'))

        self.inter_cx, self.inter_cy = cx, cy
        self.inter_road_w = road_w

    def _update_intersection(self, queues, phase):
        """Met à jour l'affichage dynamique du carrefour."""
        c = self.canvas
        c.delete('dynamic')

        cx, cy = self.inter_cx, self.inter_cy
        rw = self.inter_road_w
        max_q = self.env.max_queue if self.env else 5

        # Feux
        phase_colors = {
            0: (GREEN, RED),    # NS vert, EO rouge
            1: (RED, GREEN),    # NS rouge, EO vert
            2: (ORANGE, ORANGE) # Transition
        }
        ns_col, eo_col = phase_colors.get(phase, (GRAY, GRAY))

        # Feux N/S
        for (fx, fy) in [(cx - rw // 2 - 14, cy - 20), (cx + rw // 2 + 14, cy + 20)]:
            c.create_oval(fx - 8, fy - 8, fx + 8, fy + 8, fill=ns_col, outline='', tags='dynamic')
        # Feux E/O
        for (fx, fy) in [(cx + 20, cy - rw // 2 - 14), (cx - 20, cy + rw // 2 + 14)]:
            c.create_oval(fx - 8, fy - 8, fx + 8, fy + 8, fill=eo_col, outline='', tags='dynamic')

        # Files d'attente (barres de véhicules)
        bar_configs = [
            # direction idx, x_start, y_start, dx, dy, couleur base
            (0, cx - 12, cy - rw // 2 - 5, 0, -16, BLUE),   # Nord
            (1, cx + 2,  cy + rw // 2 + 5, 0,  16, BLUE),   # Sud
            (2, cx + rw // 2 + 5, cy - 12, 16,  0, PURPLE), # Est
            (3, cx - rw // 2 - 5, cy + 2, -16,  0, PURPLE), # Ouest
        ]
        for idx, x0, y0, dx, dy, color in bar_configs:
            q = queues[idx] if queues is not None else 0
            for v in range(q):
                xv = x0 + dx * v
                yv = y0 + dy * v
                intensity = int(180 + 75 * (v / max(max_q, 1)))
                c.create_rectangle(xv, yv, xv + 12, yv + 12,
                                   fill=color, outline='', tags='dynamic')
                c.create_text(xv + 6, yv + 6, text='🚗' if idx < 2 else '🚙',
                              font=('', 7), tags='dynamic')

        # Légende phase
        phase_labels = {0: 'Phase N/S ▶', 1: 'Phase E/O ▶', 2: '🟠 Transition'}
        phase_col = {0: GREEN, 1: GREEN, 2: ORANGE}
        c.create_text(cx, 236, text=phase_labels.get(phase, '?'),
                      fill=phase_col.get(phase, GRAY),
                      font=('Courier New', 10, 'bold'), tags='dynamic')

    # ─────────────────────── Simulation Temps Réel ───────────────────────

    def _toggle_sim(self):
        if self.simulating:
            self.simulating = False
            self.sim_btn.config(text="▶ Simuler l'agent")
        else:
            if self.agent is None and self.sim_mode.get() == 'ql':
                messagebox.showwarning("Agent non entraîné",
                                       "Veuillez d'abord entraîner l'agent Q-Learning.")
                return
            self.simulating = True
            self.sim_btn.config(text="⏸ Pause")
            self.sim_thread = threading.Thread(target=self._sim_loop, daemon=True)
            self.sim_thread.start()

    def _sim_loop(self):
        """Boucle de simulation temps réel."""
        self._init_env()
        state = self.env.reset()
        self.baseline.reset()
        total_wait = 0

        while self.simulating:
            state_idx = self.env.state_to_index(state)

            if self.sim_mode.get() == 'ql' and self.agent:
                action = self.agent.greedy_action(state_idx)
            else:
                action = self.baseline.choose_action(state_idx)

            next_state, reward, info = self.env.step(action)
            total_wait += info['waiting']
            state = next_state
            self.step_count += 1

            # Mise à jour UI (thread-safe)
            q = info['queues'].tolist()
            ph = info['phase']
            self.root.after(0, self._update_intersection, q, ph)
            self.root.after(0, self.stat_vars['wait_var'].set,
                            str(total_wait))

            import time
            time.sleep(0.25)

    # ─────────────────────── Entraînement ───────────────────────

    def _start_training(self):
        if self.training:
            return
        self.training = True
        self.train_btn.config(state='disabled', text='Entraînement...')
        self.ql_rewards = []
        self.base_rewards = []
        self.episode_count = 0
        self.train_thread = threading.Thread(target=self._train_loop, daemon=True)
        self.train_thread.start()

    def _train_loop(self):
        scenario = self.scenario_var.get()
        n_ep = int(self.episodes_var.get())
        alpha = float(self.alpha_var.get())
        gamma = float(self.gamma_var.get())
        e_decay = float(self.edecay_var.get())

        cfg = SCENARIOS[scenario]
        self.env = Intersection(lambdas=cfg['lambdas'], seed=42)
        self.agent = QLearningAgent(
            n_states=self.env.n_states,
            n_actions=self.env.n_actions,
            alpha=alpha, gamma=gamma,
            epsilon=1.0, epsilon_min=0.05,
            epsilon_decay=e_decay, seed=42,
        )
        baseline = FixedCycleBaseline(period=5)

        for ep in range(n_ep):
            if not self.training:
                break

            r_ql = run_episode(self.env, self.agent, n_steps=200, train=True)
            self.agent.decay_epsilon()
            self.ql_rewards.append(r_ql)

            env_b = Intersection(lambdas=cfg['lambdas'], seed=42 + ep)
            r_base = run_baseline_episode(env_b, baseline, n_steps=200)
            self.base_rewards.append(r_base)

            self.episode_count = ep + 1
            progress = (ep + 1) / n_ep * 100
            eps = self.agent.epsilon

            # Update UI every 5 episodes
            if (ep + 1) % 5 == 0:
                ql_mean = np.mean(self.ql_rewards[-20:]) if len(self.ql_rewards) >= 20 else np.mean(self.ql_rewards)
                base_mean = np.mean(self.base_rewards[-20:]) if len(self.base_rewards) >= 20 else np.mean(self.base_rewards)
                self.root.after(0, self._update_training_ui,
                                ep + 1, eps, ql_mean, base_mean, progress)

        self.root.after(0, self._training_done)

    def _update_training_ui(self, ep, eps, ql_mean, base_mean, progress):
        self.stat_vars['ep_var'].set(str(ep))
        self.stat_vars['eps_var'].set(f'{eps:.3f}')
        self.stat_vars['rew_var'].set(f'{ql_mean:.1f}')
        self.stat_vars['base_var'].set(f'{base_mean:.1f}')
        self.progress_var.set(progress)
        self.status_var.set(f"Épisode {ep} | ε = {eps:.3f} | Récompense moy. = {ql_mean:.1f}")
        self._update_reward_plot()
        self._update_q_heatmap_plot()

    def _training_done(self):
        self.training = False
        self.train_btn.config(state='normal', text='▶ Entraîner')
        self.progress_var.set(100)
        n = len(self.ql_rewards)
        ql_f = np.mean(self.ql_rewards[-50:])
        base_f = np.mean(self.base_rewards[-50:])
        self.status_var.set(
            f"✅ Terminé ({n} épisodes) | Q-Learning : {ql_f:.1f} | Baseline : {base_f:.1f}"
        )
        self._update_reward_plot()
        self._update_q_heatmap_plot()
        # Sauvegarder figures
        self._save_figures()

    # ─────────────────────── Graphes ───────────────────────

    def _smooth(self, data, w=15):
        if len(data) < w:
            return list(range(len(data))), data
        smoothed = np.convolve(data, np.ones(w) / w, mode='valid')
        return list(range(w - 1, len(data))), smoothed.tolist()

    def _update_reward_plot(self):
        ax = self.ax_reward
        ax.cla()
        ax.set_facecolor(PANEL)
        ax.set_title('Récompense totale par épisode', fontsize=10, color=WHITE)
        ax.set_xlabel('Épisode', color=GRAY)
        ax.set_ylabel('Récompense', color=GRAY)
        ax.tick_params(colors=GRAY, labelsize=8)
        ax.grid(True, color=ACCENT, alpha=0.5, linewidth=0.5)

        if self.ql_rewards:
            x = list(range(len(self.ql_rewards)))
            ax.plot(x, self.ql_rewards, alpha=0.2, color=BLUE, linewidth=0.7)
            sx, sy = self._smooth(self.ql_rewards)
            ax.plot(sx, sy, color=BLUE, linewidth=2, label='Q-Learning')

        if self.base_rewards:
            x = list(range(len(self.base_rewards)))
            ax.plot(x, self.base_rewards, alpha=0.2, color=RED, linewidth=0.7)
            sx, sy = self._smooth(self.base_rewards)
            ax.plot(sx, sy, color=RED, linewidth=2, label='Baseline fixe')

        ax.legend(fontsize=8, facecolor=PANEL, labelcolor=WHITE, edgecolor=ACCENT)
        for spine in ax.spines.values():
            spine.set_color(ACCENT)
        self.chart_canvas.draw_idle()

    def _update_q_heatmap_plot(self):
        ax = self.ax_q
        ax.cla()
        ax.set_facecolor(PANEL)
        ax.set_title('Table Q — Action optimale (q_E=0, q_O=0, phase N/S)', fontsize=10, color=WHITE)
        ax.set_xlabel('q_S', color=GRAY)
        ax.set_ylabel('q_N', color=GRAY)
        ax.tick_params(colors=GRAY, labelsize=8)

        if self.agent is None or self.env is None:
            ax.text(0.5, 0.5, "Entraîner l'agent pour voir la table Q",
                    ha='center', va='center', transform=ax.transAxes,
                    color=GRAY, fontsize=10)
            self.chart_canvas.draw_idle()
            return

        max_q = self.env.max_queue
        Z = np.zeros((max_q + 1, max_q + 1))
        for qN in range(max_q + 1):
            for qS in range(max_q + 1):
                state = (qN, qS, 0, 0, 0)  # phase N/S
                idx = self.env.state_to_index(state)
                Z[qN, qS] = np.argmax(self.agent.Q[idx])

        im = ax.imshow(Z, origin='lower', cmap='RdYlGn', aspect='auto',
                       vmin=0, vmax=1, interpolation='nearest')
        ax.set_xticks(range(max_q + 1))
        ax.set_yticks(range(max_q + 1))

        from matplotlib.patches import Patch
        legend = [Patch(facecolor='#4ade80', label='0 = Maintenir'),
                  Patch(facecolor='#f87171', label='1 = Changer')]
        ax.legend(handles=legend, fontsize=8, facecolor=PANEL,
                  labelcolor=WHITE, edgecolor=ACCENT, loc='upper right')
        for spine in ax.spines.values():
            spine.set_color(ACCENT)

        self.fig.colorbar(im, ax=ax)
        self.chart_canvas.draw_idle()

    def _save_figures(self):
        """Sauvegarde les figures en PNG."""
        try:
            os.makedirs('figures', exist_ok=True)
            sc = self.scenario_var.get()

            # Convergence
            fig2, ax = plt.subplots(figsize=(9, 4))
            fig2.patch.set_facecolor(BG)
            ax.set_facecolor(PANEL)
            if self.ql_rewards:
                x = list(range(len(self.ql_rewards)))
                ax.plot(x, self.ql_rewards, alpha=0.2, color=BLUE, linewidth=0.7)
                sx, sy = self._smooth(self.ql_rewards, 20)
                ax.plot(sx, sy, color=BLUE, linewidth=2, label='Q-Learning')
            if self.base_rewards:
                x = list(range(len(self.base_rewards)))
                ax.plot(x, self.base_rewards, alpha=0.2, color=RED, linewidth=0.7)
                sx, sy = self._smooth(self.base_rewards, 20)
                ax.plot(sx, sy, color=RED, linewidth=2, label='Baseline fixe')
            ax.set_title(f'Convergence — {SCENARIOS[sc]["label"]}', color=WHITE)
            ax.set_xlabel('Épisode', color=GRAY)
            ax.set_ylabel('Récompense', color=GRAY)
            ax.tick_params(colors=GRAY)
            ax.legend(facecolor=PANEL, labelcolor=WHITE, edgecolor=ACCENT)
            ax.grid(True, color=ACCENT, alpha=0.4)
            plt.tight_layout()
            plt.savefig(f'figures/convergence_{sc}.png', dpi=150, bbox_inches='tight',
                        facecolor=BG)
            plt.close(fig2)
            print(f"Figures sauvegardées dans figures/")
        except Exception as e:
            print(f"Erreur sauvegarde figures : {e}")

    # ─────────────────────── Utils ───────────────────────

    def _init_env(self, *args):
        scenario = self.scenario_var.get()
        cfg = SCENARIOS[scenario]
        self.env = Intersection(lambdas=cfg['lambdas'], seed=42)
        self._update_intersection([0, 0, 0, 0], 0)

    def _reset_all(self):
        self.training = False
        self.simulating = False
        self.agent = None
        self.ql_rewards = []
        self.base_rewards = []
        self.episode_count = 0
        self.progress_var.set(0)
        self.status_var.set("Réinitialisé.")
        for k in self.stat_vars:
            self.stat_vars[k].set('—' if 'var' in k else '0')
        self.train_btn.config(state='normal', text='▶ Entraîner')
        self._init_env()
        self._update_reward_plot()
        self._update_q_heatmap_plot()


def main():
    root = tk.Tk()
    app = TrafficGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
