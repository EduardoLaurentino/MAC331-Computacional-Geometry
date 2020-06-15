#!/usr/bin/env python

from geocomp.common.point import Point

class Point(Point):
	"""Um ponto representado por suas coordenadas cartesianas"""
	def __init__(self, x, y):
		super().__init__(x, y)
	def __lt__(self, other):
		return self.y < other.y or (self.y == other.y and self.x > other.x)

	def __gt__(self, other):
		return self.y > other.y or (self.y == other.y and self.x < other.x)

	def __str__(self):
		return f'({self.x}, {self.y})'
