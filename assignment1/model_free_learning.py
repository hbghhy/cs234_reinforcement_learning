### Episode model free learning using Q-learning and SARSA

# Do not change the arguments and output types of any of the functions provided! You may debug in Main and elsewhere.

import numpy as np
import gym
import time
from lake_envs import *
import tqdm



def learn_Q_QLearning(env, num_episodes=2000, gamma=0.95, lr=0.1, e=0.8, decay_rate=0.99):
    """Learn state-action values using the Q-learning algorithm with epsilon-greedy exploration strategy.
    Update Q at the end of every episode.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to compute Q function for. Must have nS, nA, and P as
      attributes.
    num_episodes: int
      Number of episodes of training.
    gamma: float
      Discount factor. Number in range [0, 1)
    learning_rate: float
      Learning rate. Number in range [0, 1)
    e: float
      Epsilon value used in the epsilon-greedy method.
    decay_rate: float
      Rate at which epsilon falls. Number in range [0, 1)

    Returns
    -------
    np.array
      An array of shape [env.nS x env.nA] representing state, action values
    """

    ############################
    # YOUR IMPLEMENTATION HERE #
    Q = np.zeros((env.nS, env.nA))
    temp_e = e
    for _ in range(num_episodes):
        s = 0
        e = temp_e
        while True:
            if np.random.random() < e:
                a = np.random.choice(list(env.P[s].keys()), 1)[0]
            else:
                a = np.argmax(Q[s])
            r = np.random.random()
            for probability, nextstate, reward, terminal in env.P[s][a]:
                if r <= probability:
                    Q[s][a] += lr * (reward + gamma * np.max(Q[nextstate]) - Q[s][a])
                    s = nextstate
                    break
                else:
                    r -= probability
            e *= decay_rate
            if terminal:
                break
    ############################
    return Q


def learn_Q_SARSA(env, num_episodes=2000, gamma=0.95, lr=0.1, e=0.8, decay_rate=0.99):
    """Learn state-action values using the SARSA algorithm with epsilon-greedy exploration strategy
    Update Q at the end of every episode.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to compute Q function for. Must have nS, nA, and P as
      attributes.
    num_episodes: int
      Number of episodes of training.
    gamma: float
      Discount factor. Number in range [0, 1)
    learning_rate: float
      Learning rate. Number in range [0, 1)
    e: float
      Epsilon value used in the epsilon-greedy method.
    decay_rate: float
      Rate at which epsilon falls. Number in range [0, 1)

    Returns
    -------
    np.array
      An array of shape [env.nS x env.nA] representing state-action values
    """

    ############################
    # YOUR IMPLEMENTATION HERE #
    Q = np.zeros((env.nS, env.nA))
    temp_e = e
    for _ in range(num_episodes):
        s = 0
        e = temp_e
        if np.random.random() < e:
            a = np.random.choice(list(env.P[s].keys()), 1)[0]
        else:
            a = np.argmax(Q[s])
        while True:
            r = np.random.random()
            for probability, nextstate, reward, terminal in env.P[s][a]:
                if r <= probability:
                    if np.random.random() < e:
                        a1 = np.random.choice(list(env.P[nextstate].keys()), 1)[0]
                    else:
                        a1 = np.argmax(Q[nextstate])
                    Q[s][a] += lr * (reward + gamma * Q[nextstate][a1] - Q[s][a])
                    s = nextstate
                    a = a1
                    break
                else:
                    r -= probability
            e *= decay_rate
            if terminal:
                break
    ############################

    return Q


def render_single_Q(env, Q):
    """Renders Q function once on environment. Watch your agent play!

      Parameters
      ----------
      env: gym.core.Environment
        Environment to play Q function on. Must have nS, nA, and P as
        attributes.
      Q: np.array of shape [env.nS x env.nA]
        state-action values.
    """

    episode_reward = 0
    state = env.reset()
    done = False
    while not done:
        env.render()
        time.sleep(0.5)  # Seconds between frames. Modify as you wish.
        action = np.argmax(Q[state])
        state, reward, done, _ = env.step(action)
        episode_reward += reward

    print("Episode reward: %f" % episode_reward)
    return episode_reward

def evaluate_q(env,Q):

    episode_reward = 0
    state = env.reset()
    done = False
    while not done:
        #env.render()
        action = np.argmax(Q[state])
        state, reward, done, _ = env.step(action)
        episode_reward += reward
    return episode_reward

def average_score(env,Q,time=100):
    return np.mean([evaluate_q(env,Q) for _ in range(time)])

def evaluate_type(env,num_episodes,type):
    if type=='q-learning':
        fun=learn_Q_QLearning
    elif type=='sarsa':
        fun=learn_Q_SARSA
    else:
        raise ValueError
    ans=np.zeros(num_episodes.shape)
    for i in tqdm.tqdm(range(num_episodes.shape[0])):
        Q=fun(env,num_episodes[i])
        ans[i]=average_score(env,Q)
    return ans


# Feel free to run your own debug code in main!
def main():
    env = gym.make('Stochastic-4x4-FrozenLake-v0')
    num_episodes=np.arange(1,2000,5)
    q_out1=evaluate_type(env,num_episodes,'q-learning')
    q_out2=evaluate_type(env,num_episodes,'sarsa')
    import matplotlib.pyplot as plt

    plt.plot(num_episodes, q_out1)
    plt.plot(num_episodes, q_out2)

    plt.legend(['q-learning', 'sarsa'], loc='upper left')

    plt.show()


if __name__ == '__main__':
    main()
