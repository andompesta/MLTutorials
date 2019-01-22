# Double DQN

The traditional DQN implementation tend to overestimate the Q-values.
The problem is that initially we are not sure to take, in  the next-state, the the action with the highest Q-value.
Because the Q-values depend on the training done so far.
Therefore, taking the maximum Q-value (which is noisy) as the best action to take can lead to false positives. 
If non-optimal actions are regularly given a higher Q value than the optimal best action, the learning will be complicated.

Thus, we use two networks to decouple the action selection from the target Q value generation. 
We:
- use our DQN network to select what is the best action to take for the next state (the action with the highest Q value)
- use our target network to calculate the target Q value of taking that action at the next state.

Formally, the TD-error is computed as:
<p align="center">
    <img src="./figures/double_dqn.png" width="900px" height="190px"/>
    <br />
    <a name="eq-deep_q_learning_update"> Eq. 1: TD-error computed in the Double DQN</a>
</p>

# Dueling DQN
