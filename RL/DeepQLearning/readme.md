# PyTorch DQN

A pytorch implementation of the DQN algorithm proposed in http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html?foxtrotcallback=true

All the code is based on @dennybritz RL tutorial (https://github.com/dennybritz/reinforcement-learning)


## Requrements
All the requred library are in the env.yml files.
Gym is vizdoom is required to run the enviroment's simulator.

## Deep Q-learning

Instead of using a table, we employ a neural network to learn the expected future of an action.
Namely, Doom is a complex environment; thus it is impractical to have a table for each possible state.
[Fig 1.](#fig-deep_q_learning-vs-q_learning) represents the difference between Q-learning and Deep Q-learning where a neural network is used to approximate the q-values.


<p align="center">
    <img src="./figures/deep_q_learning.png" width="600px" height="900px"/>
    <br />
    <a name="fig-deep_q_learning-vs-q_learning"> Fig. 1: Deep Q-learning vs. Q-learning</a>
</p>

