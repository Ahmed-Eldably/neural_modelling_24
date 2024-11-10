# Neural Modelling Winter 2024 Course - University of Tübingen

## Project Overview

This project simulates various classical conditioning paradigms using the Rescorla-Wagner (RW) model, a foundational model in computational neuroscience for learning associations between stimuli and rewards. This project explores how the RW model performs across different paradigms: Blocking, Inhibitory Conditioning, Overshadowing, Secondary Conditioning, and Explaining Away. Additionally, we apply the RW model to examine an Extinction paradigm.
## Table of Contents
1. [Project Overview](#project-overview)
2. [Implemented Paradigms](#implemented-paradigms)
3. [Setup and Installation](#setup-and-installation)
4. [Usage](#usage)
5. [References](#references)

## Implemented Paradigms

### 1. Blocking
   - **Description**: Tests whether a previously learned association with a stimulus (e.g., S1) blocks a new stimulus (S2) from forming a similar association.
   - **Expected Outcome**: S1 should maintain a strong association, while S2 remains weakly associated or not at all.

###  2. Inhibitory Conditioning
   - **Description**: Tests the model's ability to learn that S2 predicts the absence of a reward.
   - **Expected Outcome**: S1 should be positively associated with the reward, while S2 should indicate the absence of the reward.

### 3. Overshadowing
   - **Description**: Examines how a stronger stimulus (S1) can overshadow a less salient one (S2) when both are presented together.
   - **Expected Outcome**: S1 should dominate in associative strength, with S2 showing a lesser association.

### 4. Secondary Conditioning
   - **Description**: Explores indirect association, where S2 should form an association with the reward via S1.
   - **Expected Outcome**: Ideally, S2 should develop an indirect association with the reward.

### 5. Explaining Away
   - **Description**: Tests how an animal adjusts its expectations when one stimulus is absent.
   - **Expected Outcome**: Initial associations with both stimuli are strong, but expectations decrease when one stimulus is absent.

### 6. Extinction Paradigm
   - **Description**: Involves conditioning a stimulus (S1) to a reward, followed by extinction trials with no reward, and then a delayed test.
   - **Expected Outcome**: The model should capture a decrease in the association during extinction, with potential recovery during the test.

## Setup and Installation

### Prerequisites
- Python 3.10+
- Required packages: `numpy`, `matplotlib`

### Installation
1. Clone the repository:
   ```bash
   git clone git@github.com:Ahmed-Eldably/neural_modelling_24.git
   cd neural modelling_24
   ```
   
## Usage

Still, work in progress to simulate all paradigms from one file. For now, running each paradigm in 02_conditioning or 03_extinction should be enough to get the graphs.
Also, the plots are already saved in each folder.

## References
   * Wilson, W. J. (2012). The Rescorla-Wagner Model, Simplified. Albion College. Disponible en:< http://webcache. googleusercontent. com/search.
   * Rescorla, R.A., 1972. A theory of Pavlovian conditioning: Variations in the effectiveness of reinforcement and non-reinforcement. Classical conditioning, Current research and theory, 2, pp.64-69.
   * Dayan, P., & Abbott, L. F. (2005). Theoretical neuroscience: computational and mathematical modeling of neural systems. MIT press.
