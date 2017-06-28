### MDP Value Iteration and Policy Iteratoin
# You might not need to use all parameters

import numpy as np
import gym
import time
from lake_envs import *
import random

np.set_printoptions(precision=3)


def convergence(V, V_new, tol):
    return np.all(np.abs(V - V_new) < tol)


def value_iteration(P, nS, nA, gamma=0.9, max_iteration=200, tol=1e-3):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.

    Parameters:
    ----------
    P: dictionary
        It is from gym.core.Environment
        P[state][action] is tuples with (probability, nextstate, reward, terminal)
    nS: int
        number of states
    nA: int
        number of actions
    gamma: float
        Discount factor. Number in range [0, 1)
    max_iteration: int
        The maximum number of iterations to run before stopping. Feel free to change it.
    tol: float
        Determines when value function has converged.
    Returns:
    ----------
    value function: np.ndarray
    policy: np.ndarray
    """
    V = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    ############################
    # YOUR IMPLEMENTATION HERE #

    for i in range(max_iteration):
        V_next = np.zeros(nS, dtype=float)
        for s in range(nS):
            for a in range(nA):
                r = np.random.random()
                for probability, nextstate, reward, terminal in env.P[s][a]:
                    if r <= probability:
                        V_next[s] = np.max([V_next[s],(reward + gamma * V[nextstate])])
                        break
                    else:
                        r -= probability
        if convergence(V,V_next,tol):break
        V=V_next

    for s in range(nS):
        for a in range(nA):
            for probability, nextstate, reward, terminal in P[s][a]:
                if V[nextstate]>V[policy[s]]:
                    policy[s]=a
    ############################
    return V, policy


def policy_evaluation(P, nS, nA, policy, gamma=0.9, max_iteration=1000, tol=1e-3):
    """Evaluate the value function from a given policy.

    Parameters
    ----------
    P: dictionary
        It is from gym.core.Environment
        P[state][action] is tuples with (probability, nextstate, reward, terminal)
    nS: int
        number of states
    nA: int
        number of actions
    gamma: float
        Discount factor. Number in range [0, 1)
    policy: np.array
        The policy to evaluate. Maps states to actions.
    max_iteration: int
        The maximum number of iterations to run before stopping. Feel free to change it.
    tol: float
        Determines when value function has converged.
    Returns
    -------
    value function: np.ndarray
        The value function from the given policy.
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    V = np.zeros(nS, dtype='float')
    for i in range(max_iteration):
        V_next = np.zeros(nS, dtype='float')
        for s in range(nS):
            r = np.random.random()
            for probability, nextstate, reward, terminal in env.P[s][policy[s]]:
                if r <=probability:
                    V_next[s]=reward+gamma*V[nextstate]
                    break
                else:
                    r -= probability
        if convergence(V, V_next, tol):
            break
        V = V_next
    ############################
    return V


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
    """Given the value function from policy improve the policy.

    Parameters
    ----------
    P: dictionary
        It is from gym.core.Environment
        P[state][action] is tuples with (probability, nextstate, reward, terminal)
    nS: int
        number of states
    nA: int
        number of actions
    gamma: float
        Discount factor. Number in range [0, 1)
    value_from_policy: np.ndarray
        The value calculated from the policy
    policy: np.array
        The previous policy.

    Returns
    -------
    new policy: np.ndarray
        An array of integers. Each integer is the optimal action to take
        in that state according to the environment dynamics and the
        given value function.
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    policy_next = np.zeros(nS, dtype='int')
    for s in range(nS):
        Q = np.zeros(nA, dtype='float')
        q = -np.inf
        for a in range(nA):
            temp = 0.
            rewards = 0.
            probabilities = 0.
            for probability, nextstate, reward, terminal in P[s][policy[s]]:
                temp += probability * value_from_policy[nextstate]
                rewards += reward * probability
                probabilities += probability
            Q[a] = (rewards + gamma * temp) / probabilities
            if Q[a] > q:
                q = Q[a]
                policy_next[s] = a
            elif Q[a] == q:
                if random.random() < 0.5:
                    policy_next[s] = a
    ############################
    return policy_next


def policy_iteration(P, nS, nA, gamma=0.9, max_iteration=20, tol=1e-3):
    """Runs policy iteration.

    You should use the policy_evaluation and policy_improvement methods to
    implement this method.

    Parameters
    ----------
    P: dictionary
        It is from gym.core.Environment
        P[state][action] is tuples with (probability, nextstate, reward, terminal)
    nS: int
        number of states
    nA: int
        number of actions
    gamma: float
        Discount factor. Number in range [0, 1)
    max_iteration: int
        The maximum number of iterations to run before stopping. Feel free to change it.
    tol: float
        Determines when value function has converged.
    Returns:
    ----------
    value function: np.ndarray
    policy: np.ndarray
    """
    V = np.zeros(nS, dtype='float')
    policy = np.zeros(nS, dtype='int')
    ############################
    # YOUR IMPLEMENTATION HERE #

    for s in range(nS):
        policy[s] = random.choice(list(P[s].keys()))

    for i in range(max_iteration):
        value_from_policy = policy_evaluation(P, nS, nA, policy, gamma)
        policy_next = policy_improvement(P, nS, nA, value_from_policy, policy, gamma)
        if convergence(policy, policy_next, tol):
            break
        policy = policy_next
    ############################
    return V, policy


def example(env):
    """Show an example of gym
    Parameters
        ----------
        env: gym.core.Environment
            Environment to play on. Must have nS, nA, and P as
            attributes.
    """
    env.seed(0);
    from gym.spaces import prng;
    prng.seed(10)  # for print the location
    # Generate the episode
    ob = env.reset()
    for t in range(100):
        env.render()
        a = env.action_space.sample()
        ob, rew, done, _ = env.step(a)
        if done:
            break
    assert done
    env.render();


def render_single(env, policy):
    """Renders policy once on environment. Watch your agent play!

        Parameters
        ----------
        env: gym.core.Environment
            Environment to play on. Must have nS, nA, and P as
            attributes.
        Policy: np.array of shape [env.nS]
            The action to take at a given state
    """

    episode_reward = 0
    ob = env.reset()
    for t in range(100):
        env.render()
        time.sleep(0.5)  # Seconds between frames. Modify as you wish.
        a = policy[ob]
        ob, rew, done, _ = env.step(a)
        episode_reward += rew
        if done:
            break
    assert done
    env.render();
    print
    "Episode reward: %f" % episode_reward


# Part 4c
def part4c():
    print('Part 4c\n--------')
    env_names = ('Deterministic-4x4-FrozenLake-v0', 'Stochastic-4x4-FrozenLake-v0')
    for env_name in env_names[1:]:
        env = gym.make(env_name)
        # print(env.__doc__)
        # print("Here is an example of state, action, reward, and next state")
        # example(env)
        gamma = 0.8
        # V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=gamma, max_iteration=300, tol=1e-5)
        V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=gamma, max_iteration=300, tol=1e-5)

        print('Environment: "%s"' % env_name)
        print('------------------------------')
        print('Policy Iteration')
        print('  Optimal Value Function: %r' % V_pi)
        print('  Optimal Policy:         %r' % p_pi)
        print('Value Iteration')
        print('  Optimal Value Function: %r' % V_vi)
        print('  Optimal Policy:         %r' % p_vi)
        print('\n##########\n##########\n\n')
        render_single(env, p_pi)
        print('\n\n\n\n')
        # render_single(env, p_vi)


# Feel free to run your own debug code in main!
# Play around with these hyperparameters.
if __name__ == "__main__":
    env = gym.make("Deterministic-4x4-FrozenLake-v0")
    print
    env.__doc__
    print
    "Here is an example of state, action, reward, and next state"
    example(env)
    V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, max_iteration=20, tol=1e-3)
    V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, max_iteration=20, tol=1e-3)
    print('Policy Iteration')
    print('  Optimal Value Function: %r' % V_pi)
    print('  Optimal Policy:         %r' % p_pi)
    print('Value Iteration')
    print('  Optimal Value Function: %r' % V_vi)
    print('  Optimal Policy:         %r' % p_vi)
    print('\n##########\n##########\n\n')
    #render_single(env, p_vi)
    #part4c()
