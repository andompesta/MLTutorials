# Value Function RL
In Value-Based RL we use the advante function to define our policy.

According to it's definition, the advantage function <a href="https://www.codecogs.com/eqnedit.php?latex=A^{\pi}(s_t,&space;a_t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?A^{\pi}(s_t,&space;a_t)" title="A^{\pi}(s_t, a_t)" /></a> tell us how much better <a href="https://www.codecogs.com/eqnedit.php?latex=a_t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a_t" title="a_t" /></a> is w.r.t. the average action according to policy <a href="https://www.codecogs.com/eqnedit.php?latex=\pi" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\pi"/></a>.

<img src="https://latex.codecogs.com/gif.latex?arg&space;\max_{a_t}&space;A^{\pi}(s_t,&space;a_t)" title="arg \max_{a_t} A^{\pi}(s_t, a_t)" />: best action from <img src="https://latex.codecogs.com/gif.latex?s_t"/>, if we follow <img src="https://latex.codecogs.com/gif.latex?\pi"/>. This is at least as good as <img src="https://latex.codecogs.com/gif.latex?a_t\sim\pi(a_t|s_t)" title="a_t \sim \pi(a_t | s_t)" /> regardless of <img src="https://latex.codecogs.com/gif.latex?\pi(a_t|s_t)" title="\pi(a_t | s_t)" />.
As shown in [Fig. 1](#fig-v-function-learning), it is possible to use such arg max formulation of the advantage function to define our policy.


<p align="center">
    <img src="./figures/v-function-learning-process.png" width="400px" height="300px"/>
    <br />
    <a name="fig-v-function-learning"> Fig. 1: Value function learning process.</a>
</p>


In real-world scenarios we have to represent the value function <img src="https://latex.codecogs.com/gif.latex?V^{\pi}(s_t)" /> as a neural network because it has to scale to every state.

<p align="center">
    <img src="./figures/value-function-approx.png" width="500px" height="250px"/>
    <br />
</p>

Then we can define a training loss function as: <img src="https://latex.codecogs.com/gif.latex?\mathcal{L}(\theta)&space;=&space;\frac{1}{2}&space;||&space;V_{\theta}(s)&space;-&space;\max_{a_t}&space;Q(s,a)||^2" title="\mathcal{L}(\theta) = \frac{1}{2} || V_{\theta}(s) - \max_{a_t} Q(s,a)||^2" />.
Finally we can use a fitted value iteration algorithm:
1. <img src="https://latex.codecogs.com/gif.latex?y_t&space;=&space;max_{a_t}\big&space;(&space;r(s_t,&space;a_t)&space;&plus;&space;\gamma&space;\mathbb{E}[V_{\theta}(s_t')]&space;\big&space;)" title="y_t = max_{a_t}\big ( r(s_t, a_t) + \gamma \mathbb{E}[V_{\theta}(s_t')] \big )" />
2. <img src="https://latex.codecogs.com/gif.latex?\theta&space;=&space;arg&space;\min_{\theta}&space;\frac{1}{2}\sum_t||V_{\theta}(s_t)&space;-&space;y_t||^2" title="\theta = arg \min_{\theta} \frac{1}{2}\sum_t||V_{\theta}(s_t) - y_t||^2" />
Yet such iterative algorithm assume that we have a model ov the environment because we need to compute the value function for every action (max function) at every state. However we can't go back to previous state if we don't have a model of the environment.

It is possible to overcome such limitation using the Q-function formalism:
1. <img src="https://latex.codecogs.com/gif.latex?y_t&space;=&space;r(s_t,&space;a_t)&space;&plus;&space;\gamma&space;\mathbb{E}[V_{\theta}(s_t')]" />
2. <img src="https://latex.codecogs.com/gif.latex?\theta&space;=&space;arg&space;\min_{\theta}&space;\frac{1}{2}\sum_t||Q_{\theta}(s_t, a_t)&space;-&space;y_t||^2" title="\theta = arg \min_{\theta} \frac{1}{2}\sum_t||V_{\theta}(s_t) - y_t||^2" />
where <img src="https://latex.codecogs.com/gif.latex?\mathbb{E}[V_\theta(s_t')] \approx \max_{a_t'} Q_{\theta}(s_t', a_t') />