GridWorld Q-Learning Visualization
📌 Project Overview

This project presents a complete implementation of the Q-Learning reinforcement learning algorithm applied to a 5×5 GridWorld environment.
The agent learns an optimal policy to navigate from a start state to a goal state while avoiding obstacles and utilizing a special transition state.

The project emphasizes learning behavior analysis through rich visualizations, including state-value heatmaps, Q-tables, policy maps, and reward convergence plots under different hyperparameter settings.

🧠 Key Concepts Covered

Reinforcement Learning

Q-Learning Algorithm

Markov Decision Processes (MDP)

Epsilon-Greedy Exploration

Learning Rate (α) & Discount Factor (γ)

Policy Extraction & Evaluation

🌍 Environment Description

Grid Size: 5 × 5

Start State: (1, 0)

Goal State: (4, 4)

Obstacles: Fixed blocked cells

Special State: Teleports the agent with a positive reward

Actions

North ↑

South ↓

East →

West ←

📊 Visualizations Included

The program automatically generates the following figures:

🔹 State Value Functions

Heatmaps for learning rates:

α = 0.1

α = 0.5

α = 1.0

🔹 Q-Tables

Tabular display of Q-values for each state–action pair

🔹 Policy Visualizations

Arrow-based optimal policy for each learning rate

🔹 Reward Curves

Cumulative reward per episode

Convergence comparison across learning rates

🔹 Additional Experiments

Effect of epsilon (exploration rate)

Comparison of different learning rates

⚙️ Technologies Used

Python 3

NumPy

Matplotlib

Tabulate

▶️ How to Run

Clone the repository:

git clone https://github.com/<your-username>/gridworld-q-learning-visualization.git


Open the project directory.

Install required libraries:

pip install numpy matplotlib tabulate


Run the program:

python 6y[1].py


The program will:

Train the Q-Learning agent

Display all visualizations automatically

🧪 Experiments Performed

Learning rate comparison (α = 0.1, 0.5, 1.0)

Exploration rate (epsilon) impact analysis

Early stopping based on reward convergence

🎯 Learning Outcomes

Practical understanding of Q-Learning

Visualization of RL convergence behavior

Effect of hyperparameters on learning efficiency

Policy extraction from Q-tables

🚀 Future Improvements

Extend to larger grid sizes

Introduce stochastic transitions

Compare with SARSA and Deep Q-Learning

Save results instead of only plotting

👤 Author

Sandeep
Computer Science & Technology
Ulster University



Machine learning portfolios

Interview discussions on RL fundamentals
