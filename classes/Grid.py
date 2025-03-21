import pygame


class Grid:
	def __init__(self, x: int, y: int, width: int, height: int, grid_type: str, grid_types: list, colors: tuple):
		self.x = x
		self.y = y
		self.position = (x, y)
		self.width = width
		self.height = height
		self.colors = colors
		self.grid_color = ''
		self.grid_type = grid_type
		self.grid_types = grid_types
		self.rect = pygame.Rect(x, y, width, height)

	# self.determine_color()

	def determine_color(self):
		# print("type == "+self.grid_type)
		if self.grid_type == self.grid_types[0]:
			self.grid_color = self.colors[0]
		if self.grid_type == self.grid_types[1]:
			self.grid_color = self.colors[1]
		if self.grid_type == self.grid_types[2]:
			self.grid_color = self.colors[2]
		if self.grid_type == self.grid_types[3]:
			self.grid_color = self.colors[3]

	def set_grid_type(self, grid_type: str):
		self.grid_type = grid_type
		self.determine_color()
