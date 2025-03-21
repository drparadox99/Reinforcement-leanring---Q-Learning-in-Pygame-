import os
import pickle

import numpy as np


# Q-Learning
class QLearning:
	"""
	Q-Learning agent for the GridWorld environment.

	Args:
		env (GridWorld): The GridWorld environment.
		alpha (float): The learning rate.
		gamma (float): The discount factor.
		epsilon (float): The exploration rate.
		episodes (int): The number of episodes to train the agent.

	Attributes:
		env (GridWorld): The GridWorld environment.
		alpha (float): The learning rate.
		gamma (float): The discount factor.
		epsilon (float): The exploration rate.
		episodes (int): The number of episodes to train the agent.
		q_table (numpy.ndarray): The Q-table for the agent.

	Methods:
		choose_action: Choose an action for the agent to take.
		update_q_table: Update the Q-table based on the agent's experience.
		train: Train the agent in the environment.
		save_q_table: Save the Q-table to a file.
		load_q_table: Load the Q-table from a file.
	"""

	def __init__(self, env, alpha=0.5, gamma=0.95, epsilon=0.1, episodes=10):
		self.env = env
		self.alpha = alpha
		self.gamma = gamma
		self.epsilon = epsilon
		self.episodes = episodes
		# 3D table :Q-values state-action pair. 4 possible actions the agent can take in every state.
		self.q_table = np.zeros((self.env.grid_size, self.env.grid_size,
								 4))

	def choose_action(self, state):
		"""
		Choose an action for the agent to take.
		The agent chooses an action based on the epsilon-greedy policy.

		Args:
			state (tuple): The current state of the agent.

		Returns:
			action (int): The action the agent takes.
				0: up
				1: right
				2: down
				3: left
		"""
		if np.random.uniform(0, 1) < self.epsilon:
			return np.random.choice([0, 1, 2, 3])  # exploration
		else:
			return np.argmax(self.q_table[state])  # exploitation

	def update_q_table(self, state, action, reward, new_state):
		"""
		Update the Q-table based on the agent's experience.
		The Q-table is updated based on the Q-learning update rule.

		Args:
			state (tuple): The current state of the agent.
			action (int): The action the agent takes.
			reward (float): The reward the agent receives.
			new_state (tuple): The new state of the agent.

		Returns:
			None
		"""
		self.q_table[state][action] = ((1 - self.alpha) * self.q_table[state][action] +
									   self.alpha * (
											   reward + self.gamma * np.max(self.q_table[new_state])))

	def train(self):
		"""
		Train the agent in the environment.
		The agent is trained in the environment for a number of episodes.
		The agent's experience is stored and returned.

		Args:
			None

		Returns:
			rewards (list): The rewards the agent receives at each step.
			states (list): The states the agent visits at each step.
			starts (list): The start of each new episode. (cumulative number of states visited across episodes)
			steps_per_episode (list): The number of steps the agent takes in each episode.
		"""
		rewards = []
		states = []  # Store states at each step
		starts = []  # Store the start of each new episode (cumulative number of states visited across episodes)
		steps_per_episode = []  # Store the number of steps per episode
		steps = 0  # Initialize the step counter outside the episode loop
		episode = 0

		while episode < self.episodes:
			state = self.env.reset()
			total_reward = 0
			done = False
			while not done:
				action = self.choose_action(state)
				new_state, reward, done = self.env.step(action)
				self.update_q_table(state, action, reward, new_state)  # update after each action
				state = new_state
				total_reward += reward
				states.append(state)  # Store all state for all episodes (self.episodes)
				steps += 1  # Increment the step counter
				if done and state == self.env.goal:  # Check if the agent has reached the goal
					starts.append(len(states))  # Store the start of the new episode
					rewards.append(total_reward)
					steps_per_episode.append(steps)  # Store the number of steps for this episode
					steps = 0  # Reset the step counter
					episode += 1
		return rewards, states, starts, steps_per_episode

	def save_q_table(self, filename):
		"""
		Save the Q-table to a file.

		Args:
			filename (str): The name of the file to save the Q-table to.

		Returns:
			None
		"""
		filename = os.path.join(os.path.dirname(__file__), filename)
		with open(filename, 'wb') as f:
			pickle.dump(self.q_table, f)

	def load_q_table(self, filename):
		"""
		Load the Q-table from a file.

		Args:
			filename (str): The name of the file to load the Q-table from.

		Returns:
			None
		"""
		filename = os.path.join(os.path.dirname(__file__), filename)
		with open(filename, 'rb') as f:
			self.q_table = pickle.load(f)
