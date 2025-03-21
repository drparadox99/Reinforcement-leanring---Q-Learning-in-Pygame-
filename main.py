# logging.basicConfig(level=logging.INFO)
import os

import numpy
import pygame

from classes.Env import Env
from classes.Q_Learning import QLearning

# Game parameters
width = 500
height = 500
grid_size = 5
grid_border = 1
grid_types = ['valid_states', 'obstacle_states', 'goal_state', 'current_state']
grid_colors = ['#ffffff', '#d72828', '#90ee90', '#000000']

# Window initialization
os.environ['SDL_VIDEO_WINDOW_POS'] = '10,100'
pygame.init()
screen = pygame.display.set_mode((width, height + 25))

# Fonts initialization and settings
pygame.font.init()
pygame.display.set_caption("5x5 rl grid game")
font = pygame.font.Font(None, 33)
score_text_color = 'white'
score_surface = pygame.Surface((width, 25))  # create a custom surface with custom width and height

# Display settings
clock = pygame.time.Clock()
frames_per_sec = 10  # every second at most 10 frames should pass.
num_environments = 10


# logging.info(f"start state: {state}, goal: {env.goal}, obstacles: {env.obstacles}")
def draw_grid(grid_mat: numpy.ndarray):
	for row in range(env.grid_size):
		for col in range(env.grid_size):
			pygame.draw.rect(
				screen,
				grid_mat[row][col].grid_color,
				grid_mat[row][col].rect.inflate(-grid_border * 2, -grid_border * 2),
			)


def display_env_details(i: int, steps_per_episode: int, rewards: int):
	# Print env details (num_episodes & steps)
	print(f"Environment number {i + 1}")
	for i, steps in enumerate(steps_per_episode, 1):
		print(f"Episode {i}: {steps} steps")
	print(f"Total reward: {sum(rewards):.2f}")
	print()


for i in range(num_environments):
	print(f'Environment : {i} ')
	screen.fill('black')  # default backgroundfor

	env = Env(width, height, grid_size, grid_types, grid_colors)
	agent = QLearning(env)  # Qlearning implementation

	# Load the Q-table if it exists
	if os.path.exists(os.path.join(os.path.dirname(__file__), 'q_table.pkl')):
		agent.load_q_table('q_table.pkl')

	draw_grid(env.grid_mat)  # display grid
	pygame.display.flip()  # update the display
	rewards, states, starts, steps_per_episode = agent.train()  # Get starts and steps_per_episode as well

	# Save the Q-table
	agent.save_q_table('q_table.pkl')
	display_env_details(i, steps_per_episode, rewards)  # display env details
	num_episodes = 1
	episode_indx = 0
	episode_step = 0

	# visual depiction of actions/states of all episodes of an environement
	for idx, next_state in enumerate(states):
		env.update_grid_mat(next_state)  # update grid
		draw_grid(env.grid_mat)  # draw grid
		episode_step += 1
		if idx > sum(steps_per_episode[:episode_indx + 1]):  # episodes count
			episode_indx += 1
			num_episodes += 1
			episode_step = 0

		# create a surface for the score display
		score_display_surface = font.render(
			f'Episode: {num_episodes} | Total Rewards : {rewards[num_episodes - 1]:.2f} | Steps {episode_step} ',
			True, 'black')
		# fill custom surface
		score_surface.fill('white')
		# blit score surface onto custom score_surface
		score_surface.blit(score_display_surface, (20, 0))
		# blit custom surace (score_surface) onto main screen
		screen.blit(score_surface, (0, 0))

		pygame.display.flip()  # update the display
		clock.tick(frames_per_sec)

	# event handling
	for event in pygame.event.get():
		if event.type == pygame.quit:
			running = False

# quit pygame
pygame.quit()
