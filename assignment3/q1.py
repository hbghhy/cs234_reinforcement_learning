import math
import gym
from frozen_lake import *
import numpy as np
import time
from utils import *
import matplotlib.pyplot as plt
import tqdm


def rmax(env, gamma, m, R_max, epsilon, num_episodes, max_step=6):
    """Learn state-action values using the Rmax algorithm

    Args:
    ----------
    env: gym.core.Environment
        Environment to compute Q function for. Must have nS, nA, and P as
        attributes.
    m: int
        Threshold of visitance
    R_max: float 
        The estimated max reward that could be obtained in the game
    epsilon: 
        accuracy paramter
    num_episodes: int 
        Number of episodes of training.
    max_step: Int
        max number of steps in each episode

    Returns
    -------
    np.array
      An array of shape [env.nS x env.nA] representing state-action values
    """

    Q = np.ones((env.nS, env.nA)) * R_max / (1 - gamma)
    R = np.zeros((env.nS, env.nA))
    nSA = np.zeros((env.nS, env.nA))
    nSASP = np.zeros((env.nS, env.nA, env.nS))
    ########################################################
    #                   YOUR CODE HERE                     #
    ########################################################
    avg_scores = np.zeros(num_episodes)
    total_score = 0
    for episode in range(num_episodes):
        s = env.reset()
        done = False
        t = 0 # step
        while not done and t < max_step:
            best_a = np.argmax(Q[s])
            s_next, r, done, _ = env.step(best_a)
            total_score += r
            if nSA[s][best_a] < m:
                nSA[s][best_a] += 1
                R[s][best_a] += r
                nSASP[s][best_a][s_next] += 1
                if nSA[s][best_a] == m:              # update all Q functions
                    for i in range(int( np.ceil(np.log(1 / (epsilon * (1 - gamma))) / (1 - gamma)) )):
                        for s1 in range(env.nS):
                            for a1 in range(env.nA):
                                if nSA[s1][a1] >= m: # update Q function
                                    q = R[s1][a1] / nSA[s1][a1]
                                    for s2 in range(env.nS):
                                        transition = nSASP[s1][a1][s2] / nSA[s1][a1]
                                        q += gamma * transition * np.max(Q[s2])
                                    Q[s1][a1] = q
            s = s_next
            t += 1
        avg_scores[episode] = total_score / (episode + 1)






    ########################################################
    #                    END YOUR CODE                     #
    ########################################################
    return Q,avg_scores


def main():
    env = FrozenLakeEnv(is_slippery=False)
    print(env.__doc__)
    plt.xlabel('episodes')
    plt.ylabel('average score')
    for m in tqdm.tqdm(range(1,10,1)):
        Q,avg_scores = rmax(env, gamma=0.99, m=m, R_max=1, epsilon=0.1, num_episodes=1000)
        render_single_Q(env, Q)
        plt.plot(avg_scores)
    plt.legend(['m = '+str(i) for i in range(1,10,1)], loc='upper left')
    plt.savefig('rmax_avg_scores.png')



if __name__ == '__main__':
    print("haha")
    main()
