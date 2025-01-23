# Neural Modelling Winter 2024 - University of Tübingen

## Overview

This repository contains implementations of key reinforcement learning and neural modeling paradigms for the **Neural Modelling Winter 2024 Course**. The project explores classical and modern computational neuroscience techniques, including **Rescorla-Wagner models**, **temporal difference learning**, and **successor representation-based actor-critic frameworks**. Each module demonstrates foundational principles in learning and decision-making.

---

## Directory Structure

- **`02_prediction_learning`**: Classical conditioning paradigms using the Rescorla-Wagner model, including Blocking, Overshadowing, and Secondary Conditioning. Plots are included in the `graphs/` folder.
- **`03_td_and_successor_learning`**: Implementation of temporal difference learning and successor representation to model prediction and decision processes.
- **`04_model_fitting_and_pavlovian_biases`**: Analysis of Pavlovian-instrumental interactions and model fitting with experimental data.
- **`05_learning_how_to_act`**: Actor-critic models with static and dynamic successor representations, exploring policy learning in a maze environment.
- **`helper/`**: Utility functions, including Rescorla-Wagner model computations.

---

## Setup and Installation

### Prerequisites

- Python 3.10+
- Libraries: `numpy`, `matplotlib`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/aeldably/neural_modelling_24.git
   cd neural_modelling
   ```

2. Install the required dependencies:
   ```bash
    pip install -r requirements.txt
   ```

### Running the Code
1. Navigate to the relevant directory (e.g., 02_prediction_learning).
2. Execute scripts to generate plots and simulate paradigms:
   ```bash
   python blocking.py
   ```
For Jupyter notebooks (e.g., learning_how_to_act_nb.ipynb), open and run them cell by cell.

## Highlights
1. Rescorla-Wagner Model: Demonstrates classical conditioning paradigms like Blocking and Secondary Conditioning.
2. Temporal Difference Learning: Simulates dynamic value estimation for reward prediction.
3. Successor Representation: Enhances reinforcement learning speed and adaptability with SR-based value functions.
4. Actor-Critic Frameworks: Explores policy learning in complex environments.

## References
1. Rescorla, R.A. (1972). A theory of Pavlovian conditioning: Variations in the effectiveness of reinforcement and non-reinforcement.
2. Dayan, P., & Abbott, L. F. (2005). Theoretical neuroscience: computational and mathematical modeling of neural systems.
3. Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction.
