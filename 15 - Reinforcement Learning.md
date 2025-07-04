
# Reinforcement Learning

Reinforcement learning in inspired by research on psychology and animal learning. The problems involve an agent interacting with an environment, which provides numeric reward signals.

The model is asked to take actions with an effect on the state of the environment in order to maximize reward. 

## Markov Decision Process

_Markov Decision Process_ (MDP) is a framework used to help to make decisions on a stochastic environment. Our goal is to find a policy, which is a map that gives us all optimal actions on each state on our environment,

In order to solve MDP we use Bellam equation which divides a problem into simpler sub-problems easier to solve (Dynamic Programming).

The components of MDP include:

- _states_ $s$, beginning with initial state $s_0$
- _Actions_ $a$
- _Transition model_ $P(s'|s, a)$
- _Markov assumption_: the probability of going to $s'$ from $s$ depends only on $s$ and not on any other past actions or states
- _Reward function_ $r(s)$
- _Policy_ $\pi(s)$: the action that an agent takes in any given state

Therefore, MDP is defined by:

$$(S, A, R, P, \gamma)$$

- $S$ set of possible states
- $A$ set of possible actions
- $R$ distribution of reward given (state, action) pair
- $P$ transition probability i.e. distribution over next state given (state, action) pair
- $\gamma$ discount factor

In a loop, the agent selects an action $a_t$ and receives a reward $r_t$ and the next state $s_{t+1}$. A policy $\pi(s)$ is a function from $S$ to $A$ that specifies what action to take in each state. The objective is to find policy $\pi^*$ that maximized cumulative discounted reward $\sum_{t\ge 0}\gamma^t r(s_t)$ where $\gamma$ is the discounting factor. This controls the importance of the future rewards versus the immediate ones: It will make the agent optimize for short term or long term actions.

## Loop

In Supervised learning: given an input $x_i$ sampled from data distribution, we use the model with parameters $w$ to predict output $y$, then calculate the loss and update $w$ to reduce loss with gradient descent $w = w-\eta \nabla l(w, x_i, y_i)$. Note that the next input does not depend on previous inputs or agent prediction and loss is differentiable.

In Reinforcement Learning instead, given a state $s$, we take an action $a$ determined by policy $\pi(s)$. The environment selects next state $s'$ based on transition model $P(s'|s, a)$ and gives a reward $r(s)$ while setting the new state $s'$. Rewards are not differentiable w.r.t. model parameters and the agent's actions affect the environment and help to determine next observation.

There are two main approaches for Reinforcement Learning:

- Value-based methods: the goal of the agent is to optimize the value function $V(s)$ where the value of each state is the total amount of the reward an RL agent can expect to collect over the future from a given state.
- Policy-based approach: we define a policy which we need to optimize directly.

## Value Based Methods

### Value Function

The _value function_ gives the total amount of rewards the agent can expect from a particular state to all possible states from that state.

$$V^{\pi}(s)=\mathbb{E}[\sum_{t\ge 0}\gamma^t r(s_t)|s_0 = s, \pi]$$

The optimal value of a state is the value achievable by following the best possible policy:

$$V^*(s)=max_{\pi}\mathbb{E}[\sum_{t\ge 0}\gamma^t r(s_t)|s_0 = s, \pi]$$

It is more convenient to define the value of a state-action pair, called _Q-value function_:

$$Q^{\pi}(s, a)= \mathbb{E}[\sum_{t\ge 0}\gamma^t r(s_t)|s_0=s, a_0=a, \pi]$$

The optimal Q-value can be used to compute the optimal policy:

$$\pi^*(s)=arg\ max_aQ^*(s, a)$$

### Bellman Equation

There is a recursive relationship between optimal values of succesive states and actions:

$$Q^*(s, a)=r(s)+\gamma \sum_{s'}P(s'|s, a)\max_{a'}Q^*(s', a')$$
$$=\mathbb{E}_{s'\sim P(\cdot|s, a)}[r(s)+\gamma\ max_{a'}Q^*(s', a')|s, a]$$

if the optimal state-action values for the next time-step $Q^*(s', a')$ are known, then the optimal strategy is to take the action that maximizes the expected value.

### Algorithm

- Initialize the matrix $Q$ with zeros
- Select a random initial state
- For each episode (set of actions that starts on the initial state and ends on the goal state)
- While state is not goal state
	- Select a random possible action for the current state
	- Using this possible action consider going to this next state
	- Get maximum $Q$ value for this next state (All actions from this next state)
	- $Q^*(s, a)=r(s, a)+\gamma\ max_a[Q^*(s', a')]$

To find the optimal policy:

- se current state to the initial state
- from current state, find the action with the highest $Q$ value
- set current state equal to the previously found state
- repeat steps 2 and 3 until current state is the goal state.

The problem of this algorithm is that the state spaces are huge. To help, we can approximate Q-values using a parametric function $Q^*(s, a)\approx Q_w(s, a)$: we train a deep neural network that approximates $Q^*$.

$$Q^*(s,a)=\mathbb{E}_{s'\sim P(.|s, a)}[r(s)+\gamma\ max_{a'}Q^*(s', a')|s, a]$$

The target is:

$$y_i(s, a)=\mathbb{E}_{s'\sim P(.|s, a)}[r(s)+\gamma\ max_{a'}Q_{w_{i-1}}(s', a')|s, a]$$

The loss function would change at each iteration:

$$L_i(w_i)=\mathbb{E}_{s, a\sim \rho}[(y_i(s, a)-Q_{w_i}(s, a))^2]$$

where $\rho$ is a probability distribution over states $s$ and actions $a$ that we refer to as the _behaviour distribution_.

The gradient update then is:

$$\nabla_{w_i}L(w_i)=\mathbb{E}_{s, a\sim \rho }[(y_i(s, a)-Q_{w_i}(s, a))\nabla_{w_i}Q_{w_i}(s, a)]$$
$$= \mathbb{E}_{s, a\sim \rho, s'}[(r(s)+\gamma\ max_{a'}Q_{w_{i-1}}(s', a')-Q_{w_i}(s,a))\nabla_{w_i}Q_{w_i}(s, a)]$$

Instead of having expecatations, we sample _experiences_ $(s, a, s')$ using behaviour distribution and transition model.

## Policy Gradient Methods

We have seen that the space of the Q-value function can be very complicated. Instead of indirectly representing the policy using Q-values, it can be more efficient to parametrize $\pi$ and learn it directly.

$$\pi_{\theta}(s, a)\approx P(a|s)$$

The idea is to use a machine learning model that will learn a good policy from playing the game and receiving rewards. In particular, we need to find the best parameters $\theta$ of the policy to maximize the expected reward:

$$maximize\ J(\theta)=\mathbb{E}[\sum_{t\ge 0}\gamma^tr_t|\pi_{\theta}]=\mathbb{E}_\tau[r(\tau)],\ \tau=(s_0, a_0, r_0, s_1, a_1, r_1, ...)$$
$$=\int_{\tau}r(\tau)p(\tau; \theta)d\tau$$

Where $p(\tau; \theta)$ is the probability of trajectory $\tau$ under policy with parameters $\theta$:

$$p(\tau; \theta)=\prod_{t\ge 0}\pi_{\theta}(s_t, a_t)P(s_{t+1}|s_t, a_t)$$

We can then use gradient ascent:

$$\nabla J(\theta)=\mathbb{E}_{\tau}[r(\tau)\nabla_{\theta}\ \log\ p(\tau; \theta)]$$
$$\nabla_\theta\ log\ p(\tau; \theta)=\sum_{t\ge0}\nabla_{\theta}\log\ \pi_{\theta}(s_t, a_t)$$
Using a stochastic approximation by sampling $n$ trajectories:

$$\nabla_{\theta}J(\theta)\approx\frac{1}{N}\sum_{1}^{N}(\sum_{t=1}^{T_i}\gamma^tr_{i, t})(\sum_{t=1}^{T_i}\nabla_{\theta} \log \pi_{\theta}(s_{i, t}, a_{i, t}))$$

Therefore the steps to perform in order to optimize are:

1. Sample $N$ trajectories $\tau_i$  using current policy $\pi_{\theta}$
2. Estimate the policy gradient $\nabla_{\theta}J(\theta)$
3. Update parameters by gradient ascent $\theta \leftarrow \theta + \eta \nabla_{\theta}J(\theta)$

Intuitively, we want to go up the gradient to increase the total reward.

