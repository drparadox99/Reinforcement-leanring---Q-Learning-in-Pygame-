import logging
import random

import numpy as np

logging.basicConfig(level=logging.INFO)
from classes.Grid import Grid


class Env:
	"""
	GridWorld environment for navigation.

	Args:
	- width: Width of the grid
	- height: Height of the grid
	- start: Start position of the agent
	- goal: Goal position of the agent
	- obstacles: List of obstacles in the grid

	Methods:
	- reset: Reset the environment to the start state
	- is_valid_state: Check if the given state is valid
	- step: Take a step in the environment
	"""

	def __init__(
			self,
			width: int,
			height: int,
			grid_size: int,
			grid_types: list,
			grid_colors: list,
	):
		self.width = width
		self.height = height
		self.cell_width = width // grid_size
		self.cell_height = height // grid_size
		self.grid_size = grid_size
		self.grid_types = grid_types
		self.grid_colors = grid_colors

		# possible actions
		self.actions = {
			"up": np.array([-1, 0]),
			"down": np.array([1, 0]),
			"left": np.array([0, -1]),
			"right": np.array([0, 1]),
		}
		# get grid matrix and mapping (grid type, grid position) dict
		self.grid_mat, self.states = self.create_grid_matrix()
		# grid initial positions
		self.start = self.states['current_state'][0]
		self.goal = self.states['goal_state'][0]
		self.obstacles = self.states['obstacle_states']
		self.state = self.states['current_state'][0]
		self.valid_states = self.states['valid_states']

	def create_grid_matrix(self, obstacles_occur=0.25):
		"""
		Create env's grids and dict mapping grids' type to their position in the grid matrix

		Returns:
		- grid_mat : Grid matrix
		- states : Mapping dictionary (type, position)
		"""
		# number of occurrence of each state
		num_grids = self.grid_size * self.grid_size
		num_current_states = 1
		num_goal_states = 1
		num_obstacles_states = int(obstacles_occur * num_grids)
		num_valid_states = num_grids - (
				num_current_states + num_goal_states + num_obstacles_states
		)

		# determine occurence of grid_types
		grid_types_occurr = [self.grid_types[0] for i in range(num_valid_states)]
		grid_types_occurr = grid_types_occurr + [self.grid_types[1] for i in range(num_obstacles_states)]
		random.shuffle(grid_types_occurr)
		grid_types_occurr.insert(0, self.grid_types[3])
		grid_types_occurr = grid_types_occurr + [self.grid_types[2] for i in range(num_goal_states)]

		# shuffle list of grid_types

		# grid dict mapping tupe to position in the matrix
		states = {
			self.grid_types[0]: [],
			self.grid_types[1]: [],
			self.grid_types[2]: [],
			self.grid_types[3]: []
		}

		# create the grid martix
		grid_mat = []
		for row in range(self.grid_size):
			grid_row = []
			for col in range(self.grid_size):
				grid_type_temp = grid_types_occurr.pop(0)
				states[grid_type_temp].append((row, col))

				grid = Grid(
					col * self.cell_width,  # x starting point
					row * self.cell_height + 25,  # y starting point
					self.cell_width,
					self.cell_height,
					grid_type_temp,
					self.grid_types,
					self.grid_colors,
				)
				# paint grid with appropriate color
				grid.determine_color()
				grid_row.append(grid)
			grid_mat.append(grid_row)
		grid_mat = np.array(grid_mat, dtype=object)
		return grid_mat, states

	def reset(self):
		"""
		Reset the environment to the start state

		Returns:
		- Start state of the environment
		"""
		self.state = (0, 0)
		return self.state

	def is_valid_state(self, state):
		"""
		Check if the given state is valid

		Args:
		- state: State to be checked

		Returns:
		- True if the state is valid, False otherwise
		"""
		# print("state: "+ str(self.state[0]) + "<" + str(self.grid_size) )

		return (
				0 <= state[0] < self.grid_size
				and 0 <= state[1] < self.grid_size
				and all((state != obstacle).any() for obstacle in self.obstacles)
		)

	def update_grid_mat(self, current_state=None):
		"""
		Upgrade the grid matrix

		Args:
			current_state (tuple): The current state(position) of the agent.

		Returns:
			None
		"""
		for valid_state in self.valid_states:
			self.grid_mat[valid_state].set_grid_type(self.grid_types[0])
			if (self.start == (0, 0)):  # set initial position (0,0) to a valid state
				self.grid_mat[self.start].set_grid_type(self.grid_types[0])
		for obstacle in self.obstacles:
			self.grid_mat[obstacle].set_grid_type(self.grid_types[1])
		self.grid_mat[self.goal].set_grid_type(self.grid_types[2])
		if current_state is not None:
			self.grid_mat[current_state].set_grid_type(self.grid_types[3])

	def step(self, action):
		"""
		Take a step in the environment.
		The agent takes a step in the environment based on the action it chooses.

		Args:
			action (int): The action the agent takes.
				0: up
				1: right
				2: down
				3: left

		Returns:
			state (tuple): The new state of the agent.
			reward (float): The reward the agent receives.
			done (bool): Whether the episode is done or not.
		"""
		x, y = self.state
		if action == 0:  # up
			x = max(0, x - 1)
		elif action == 1:  # right
			y = min(self.grid_size - 1, y + 1)
		elif action == 2:  # down
			x = min(self.grid_size - 1, x + 1)
		elif action == 3:  # left
			y = max(0, y - 1)
		self.state = (x, y)
		if self.state in self.obstacles:
			return self.state, -1, True
		if self.state == self.goal:
			return self.state, 1, True
		return self.state, -0.01, False
