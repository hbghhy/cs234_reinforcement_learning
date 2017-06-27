import math
import gym
from frozen_lake import *
import numpy as np
import time
from utils import *

def learn_Q_QLearning(env, num_episodes=10000, gamma = 0.99, lr = 0.1, e = 0.2, max_step=6):
	"""Learn state-action values using the Q-learning algorithm with epsilon-greedy exploration strategy(no decay)
	Feel free to reuse your assignment1's code
	Parameters
	----------
	env: gym.core.Environment
		Environment to compute Q function for. Must have nS, nA, and P as attributes.
	num_episodes: int 
		Number of episodes of training.
	gamma: float
		Discount factor. Number in range [0, 1)
	learning_rate: float
		Learning rate. Number in range [0, 1)
	e: float
		Epsilon value used in the epsilon-greedy method. 
	max_step: Int
		max number of steps in each episode

	Returns
	-------
	np.array
	  An array of shape [env.nS x env.nA] representing state-action values
	"""

	Q = np.zeros((env.nS, env.nA))
	########################################################
	#                     YOUR CODE HERE                   #
	########################################################



	########################################################
	#                     END YOUR CODE                    #
	########################################################
	return Q



def main():
	env = FrozenLakeEnv(is_slippery=False)
	Q = learn_Q_QLearning(env, num_episodes = 10000, gamma = 0.99, lr = 0.1, e = 0.4)
	render_single_Q(env, Q)


if __name__ == '__main__':
	main()
