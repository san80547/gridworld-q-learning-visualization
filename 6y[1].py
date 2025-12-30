"""
GRIDWORLD Q-LEARNING — FINAL COMPLETE VERSION

Figures Produced:
Fig 1–3  : State Values (alpha = 0.1, 0.5, 1.0)
Fig 4–6  : Q-Tables (alpha = 0.1, 0.5, 1.0)
Fig 7–9  : Policy Visualizations (alpha = 0.1, 0.5, 1.0)
Fig 10–12: Reward Curves (alpha = 0.1, 0.5, 1.0)
Additional:
Epsilon Effect Graph
Learning Rate Comparison
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from tabulate import tabulate

# ===============================
# ENVIRONMENT CONFIGURATION
# ===============================

ROWS, COLS = 5, 5

START = (1, 0)
GOAL = (4, 4)
SPECIAL_FROM = (1, 3)
SPECIAL_TO = (3, 3)

OBSTACLES = {(2,2), (2,3), (2,4), (3,2)}

ACTIONS = {
    0: (-1, 0),  # North
    1: (1, 0),   # South
    2: (0, 1),   # East
    3: (0, -1),  # West
}

# ===============================
# ENVIRONMENT STEP
# ===============================

def env_step(state, action):
    r, c = state

    if state == SPECIAL_FROM:
        return SPECIAL_TO, 5, False

    dr, dc = ACTIONS[action]
    nr, nc = r + dr, c + dc

    if nr < 0 or nr >= ROWS or nc < 0 or nc >= COLS:
        return state, -1, False

    if (nr, nc) in OBSTACLES:
        return state, -1, False

    if (nr, nc) == GOAL:
        return (nr, nc), 10, True

    return (nr, nc), -1, False

def env_reset():
    return START

# ===============================
# Q-LEARNING ALGORITHM
# ===============================

def epsilon_greedy(Q, state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, 3)
    r, c = state
    return int(np.argmax(Q[r, c]))

def train_q_learning(alpha=1.0, gamma=0.99,
                     epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.98,
                     max_episodes=100, early_stop_window=30):

    Q = np.zeros((ROWS, COLS, 4))
    rewards = []
    epsilon = epsilon_start

    for episode in range(max_episodes):
        state = env_reset()
        done = False
        total_reward = 0

        for _ in range(1000):
            if done:
                break

            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, done = env_step(state, action)

            r, c = state
            nr, nc = next_state
            best_next_q = np.max(Q[nr, nc])

            Q[r, c, action] += alpha * (reward + gamma * best_next_q - Q[r, c, action])
            state = next_state
            total_reward += reward

        rewards.append(total_reward)
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        if len(rewards) >= early_stop_window and np.mean(rewards[-early_stop_window:]) > 10:
            break

    return Q, rewards

# ===============================
# VISUALIZATION FUNCTIONS
# ===============================

# 🔥 Value Heatmap with labels, numbers and black obstacles
def plot_state_values_alpha(Q, alpha):
    values = np.max(Q, axis=2)

    plt.figure(figsize=(6,5))
    plt.imshow(values, cmap="viridis")
    plt.colorbar(label="State Value")
    plt.title(f"State Values (alpha={alpha})")
    plt.xticks(range(COLS))
    plt.yticks(range(ROWS))
    plt.grid(True, linewidth=0.3)

    ax = plt.gca()

    for r in range(ROWS):
        for c in range(COLS):

            if (r, c) in OBSTACLES:
                rect = plt.Rectangle((c-0.5, r-0.5), 1, 1, color="black")
                ax.add_patch(rect)
                plt.text(c, r, "OBS", ha="center", va="center", fontsize=10, color="white", fontweight="bold")
                continue

            if (r, c) == START:
                plt.text(c, r, f"START\n{values[r,c]:.1f}", ha="center", va="center", fontsize=10, color="yellow")
                continue

            if (r, c) == GOAL:
                plt.text(c, r, f"GOAL\n{values[r,c]:.1f}", ha="center", va="center", fontsize=10, color="cyan")
                continue

            if (r, c) == SPECIAL_FROM:
                plt.text(c, r, f"JUMP\n{values[r,c]:.1f}", ha="center", va="center", fontsize=10, color="orange")
                continue

            text_color = "white" if values[r,c] < 7 else "black"
            plt.text(c, r, f"{values[r,c]:.1f}", ha="center", va="center", fontsize=10, color=text_color)

    plt.show()

# Q-table prints and display
def plot_q_table(Q, alpha):
    headers = ["State", "North", "South", "East", "West"]
    table = []

    for r in range(ROWS):
        for c in range(COLS):
            table.append([(r+1, c+1)] + list(np.round(Q[r, c], 2)))

    print("\n", tabulate(table, headers, tablefmt="grid"), "\n")

    fig, ax = plt.subplots(figsize=(7,6))
    ax.axis('off')
    t = ax.table(cellText=table, colLabels=headers, loc='center')
    t.auto_set_font_size(False)
    t.set_fontsize(7)
    plt.title(f"Q-Table (alpha={alpha})")
    plt.show()

# Policy
def plot_policy_only(Q, alpha):
    policy = np.argmax(Q, axis=2)
    arrows = {0:"↑", 1:"↓", 2:"→", 3:"←"}

    plt.figure(figsize=(6,5))
    plt.title(f"Policy (alpha={alpha})")
    plt.xlim(-0.5, COLS-0.5)
    plt.ylim(-0.5, ROWS-0.5)
    plt.gca().invert_yaxis()

    for r in range(ROWS):
        for c in range(COLS):

            if (r,c) in OBSTACLES:
                plt.text(c, r, "OBS", ha="center", color="white", fontsize=9)
                continue

            if (r,c)==START:
                plt.text(c,r,"START",ha="center",color="yellow")
                continue

            if (r,c)==GOAL:
                plt.text(c,r,"GOAL",ha="center",color="cyan")
                continue

            if (r,c)==SPECIAL_FROM:
                plt.text(c,r,"JUMP",ha="center",color="orange")
                continue

            plt.text(c, r, arrows[policy[r,c]], ha="center", fontsize=16)

    plt.grid(True)
    plt.show()

# Reward curves
def plot_reward_curve(rewards, alpha):
    plt.figure(figsize=(7,5))
    plt.plot(rewards)
    plt.title(f"Reward Curve (alpha={alpha})")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.grid(True)
    plt.show()

# Epsilon Comparison
def test_epsilon_effect(epsilon_values):
    plt.figure(figsize=(10,5))
    for eps in epsilon_values:
        Q, rewards = train_q_learning(alpha=1.0, epsilon_start=eps, epsilon_min=eps, epsilon_decay=1)
        plt.plot(rewards, label=f"eps={eps}")
    plt.title("Epsilon Impact on Learning")
    plt.legend()
    plt.grid(True)
    plt.show()

# Learning Rate Comparison
def test_learning_rate_effect(alpha_values):
    plt.figure(figsize=(10,5))
    for a in alpha_values:
        Q, rewards = train_q_learning(alpha=a)
        plt.plot(rewards, label=f"lr={a}")
    plt.title("Learning Rate Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()

# ===============================
# MAIN — RUN ALL EXPERIMENTS
# ===============================

if __name__ == "__main__":

    alphas = [0.1, 0.5, 1.0]

    for a in alphas:
        Q, rewards = train_q_learning(alpha=a)

        plot_state_values_alpha(Q, a)   # FIG 1–3
        plot_q_table(Q, a)              # FIG 4–6
        plot_policy_only(Q, a)          # FIG 7–9
        plot_reward_curve(rewards, a)   # FIG 10–12

    test_epsilon_effect([0.05, 0.1, 0.3, 0.5])
    test_learning_rate_effect([1.0, 0.8, 0.5, 0.3, 0.1])
