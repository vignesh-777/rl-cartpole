## CART POLE BALANCING
## AIM
To develop and fine-tune the Monte Carlo (MC) control algorithm to stabilize the Cart Pole.

### PROBLEM STATEMENT
The Cart Pole problem is a classic reinforcement learning task where a pole is attached to a movable cart. The goal is to apply forces (left or right) such that the pole remains upright and the cart stays within bounds.

### Key points:

Reward = +1 for each time step the pole remains upright.

### Episode ends if:

Pole angle > ±12°\
Cart position > ±2.4 units\
Objective: Maximize total reward by learning an optimal policy that keeps the pole balanced for as long as possible.

## MONTE CARLO CONTROL ALGORITHM FOR CART POLE BALANCING
The Monte Carlo (MC) control algorithm estimates the optimal action-value function Q(s, a) using returns averaged across episodes and improves the policy using ε-greedy exploration.
Steps Involved
## Initialize

Q(s, a) arbitrarily (e.g., zeros)\
π(s): ε-greedy policy w.r.t. Q(s, a)\
For each episode

Generate an episode: sequence of (state, action, reward)\
Compute returns G_t for each (s, a)\
Update Q-values [ Q(s,a) = Q(s,a) + \alpha [G_t - Q(s,a)] ]

## Policy Improvement

Make π(s) greedy w.r.t. updated Q(s, a)\
Repeat until convergence.

### MONTE CARLO CONTROL FUNCTION

```py
def mc_control (env,n_bins=g_bins, gamma = 1.0,
                init_alpha = 0.5,min_alpha = 0.01, alpha_decay_ratio = 0.5,
                init_epsilon = 1.0, min_epsilon = 0.1, epsilon_decay_ratio = 0.9,
                n_episodes = 3000, max_steps = 200, first_visit = True, init_Q=None):

    nA = env.action_space.n
    discounts = np.logspace(0, max_steps,
                            num = max_steps, base = gamma,
                            endpoint = False)
    alphas = decay_schedule(init_alpha, min_alpha,
                            0.9999, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon,
                            0.99, n_episodes)
    pi_track = []
    global Q_track
    global Q


    if init_Q is None:
        Q = np.zeros([n_bins]*env.observation_space.shape[0] + [env.action_space.n],dtype =np.float64)
    else:
        Q = init_Q

    n_elements = Q.size
    n_nonzero_elements = 0

    Q_track = np.zeros([n_episodes] + [n_bins]*env.observation_space.shape[0] + [env.action_space.n],dtype =np.float64)
    select_action = lambda state, Q, epsilon: np.argmax(Q[tuple(state)]) if np.random.random() > epsilon else np.random.randint(len(Q[tuple(state)]))

    progress_bar = tqdm(range(n_episodes), leave=False)
    steps_balanced_total = 1
    mean_steps_balanced = 0
    for e in progress_bar:
        trajectory = generate_trajectory(select_action, Q, epsilons[e],
                                    env, max_steps)

        steps_balanced_total = steps_balanced_total + len(trajectory)
        mean_steps_balanced = 0

        visited = np.zeros([n_bins]*env.observation_space.shape[0] + [env.action_space.n],dtype =np.float64)
        for t, (state, action, reward, _, _) in enumerate(trajectory):
            #if visited[tuple(state)][action] and first_visit:
            #    continue
            visited[tuple(state)][action] = True
            n_steps = len(trajectory[t:])
            G = np.sum(discounts[:n_steps]*trajectory[t:, 2])
            Q[tuple(state)][action] = Q[tuple(state)][action]+alphas[e]*(G - Q[tuple(state)][action])
        Q_track[e] = Q
        n_nonzero_elements = np.count_nonzero(Q)
        pi_track.append(np.argmax(Q, axis=env.observation_space.shape[0]))
        if e != 0:
            mean_steps_balanced = steps_balanced_total/e
        #progress_bar.set_postfix(episode=e, Epsilon=epsilons[e], Steps=f"{len(trajectory)}" ,MeanStepsBalanced=f"{mean_steps_balanced:.2f}", NonZeroValues="{0}/{1}".format(n_nonzero_elements,n_elements))
        progress_bar.set_postfix(episode=e, Epsilon=epsilons[e], StepsBalanced=f"{len(trajectory)}" ,MeanStepsBalanced=f"{mean_steps_balanced:.2f}")

    print("mean_steps_balanced={0},steps_balanced_total={1}".format(mean_steps_balanced,steps_balanced_total))
    V = np.max(Q, axis=env.observation_space.shape[0])
    pi = lambda s:{s:a for s, a in enumerate(np.argmax(Q, axis=env.observation_space.shape[0]))}[s]

    return Q, V, pi
```

### OUTPUT

```
# To run the MC control without using the previous Q values
optimal_Q, optimal_V, optimal_pi = mc_control (env,n_episodes=200)
```

<img width="642" height="42" alt="image" src="https://github.com/user-attachments/assets/740ba344-afac-4f73-9ba0-795903516ff9" />

```

# To run the MC control using the previous Q values and default parameters
optimal_Q, optimal_V, optimal_pi = mc_control (env,n_episodes=200,
                                    init_alpha = 0.5,min_alpha = 0.01, alpha_decay_ratio = 0.5,
                                    init_epsilon = 1.0, min_epsilon = 0.1, epsilon_decay_ratio = 0.9,                                               
                                    max_steps=500, init_Q=Q)
```

<img width="647" height="50" alt="image" src="https://github.com/user-attachments/assets/9625ff29-79a7-4efc-8013-32a0eb980660" />

```

optimal_Q, optimal_V, optimal_pi = mc_control (env,n_episodes=200)

```

<img width="652" height="38" alt="image" src="https://github.com/user-attachments/assets/50d660df-3782-4515-a56b-ba6cc5389171" />

```

optimal_Q, optimal_V, optimal_pi = mc_control (env,n_episodes=200,
                                    init_alpha = 0.5,min_alpha = 0.01, alpha_decay_ratio = 0.5,
                                    init_epsilon = 1.0, min_epsilon = 0.1, epsilon_decay_ratio = 0.9,
                                    max_steps=500, init_Q=Q)
```

<img width="647" height="37" alt="image" src="https://github.com/user-attachments/assets/446a75c9-179f-4ca5-a591-9a495ed0b5ef" />

```
plt.plot(x,ep1,label='ep1')
```

<img width="835" height="554" alt="image" src="https://github.com/user-attachments/assets/1ab5b79c-4d84-4e21-bbf4-8b8192ebb5a4" />


### RESULT
The Monte Carlo Control algorithm was successfully implemented for the Cart Pole environment. When initialized with pretrained Q-values, the system achieved stable pole balancing for extended durations. Thus, the Cart Pole was effectively stabilized using Monte Carlo Control.
