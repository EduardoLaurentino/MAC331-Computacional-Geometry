#!/usr/bin/env python
"""Algoritmo de Fortune"""

import math
from pqdict import pqdict
# from tkinter import *

import geocomp.gui.tk as tk
# from geocomp.gui import tk
from geocomp import config
from geocomp.common import control
from geocomp.common.guiprim import *
from geocomp.common.segment import Segment

from geocomp.voronoi.point import Point
from geocomp.voronoi.BST import *
from geocomp.voronoi.DCEL import *
from geocomp.voronoi.circumcircle import *

FORTUNE_EPS = 1e-6
FORTUNE_INF = 300
FORTUNE_PLOT_RATE = 0.05
FORTUNE_STEPS = 400

class Event():
	def __init__(self, point, is_site_event, leaf=None, center=None, face=None):
		self.point = point
		self.is_site_event = is_site_event
		self.leaf = leaf
		self.center = center
		self.face = face

	def __str__(self):
		return f'({self.point.x}, {self.point.y})'


def plot_all(leaves, V, line_y):

	par_plot = []
	for leaf in leaves:
		if leaf.pred is not None:
			pred_breakpoints = get_x_breakpoints(leaf.pred, line_y)
			startx = choose_x_breakpoint(leaf.pred, pred_breakpoints, line_y)

			bissect_line = bissect_line_function(leaf.pred)
			leaf.pred.hedge.update_origin(Point(startx, bissect_line(startx)))
		else:
			startx = -FORTUNE_INF


		if leaf.succ is not None:
			succ_breakpoints = get_x_breakpoints(leaf.succ, line_y)
			endx = choose_x_breakpoint(leaf.succ, succ_breakpoints, line_y)
			bissect_line = bissect_line_function(leaf.succ)
			leaf.succ.hedge.update_origin(Point(endx, bissect_line(endx)))
		else:
			endx = FORTUNE_INF

		par_plot += [control.plot_parabola(line_y, leaf.point.x, leaf.point.y, startx, endx, steps=FORTUNE_STEPS)]

	for h in V.hedges:
		h.segment.plot()

	sweep = control.plot_horiz_line(line_y, color='green')
	return par_plot, sweep

def unplot_all(par_plots, hedges, sweep):
	for par in par_plots:
		control.plot_delete(par)

	for h in hedges:
		if h.segment.lid is None:
			continue
		h.segment.hide()

	control.plot_delete(sweep)

def event_queue(P):
	events = [Event(Point(p.x, p.y), True, face=Face()) for p in P]
	Q = pqdict({e : e.point for e in events}, reverse=True)
	return Q

def find_borders(P):
	if len(P) == 0:
		return

	minx = miny = math.inf
	maxx = maxy = -math.inf

	for point in P:
		minx = min(point.x, minx)
		miny = min(point.y, miny)
		maxx = max(point.x, maxx)
		maxy = max(point.y, maxy)

	minx -= 0.2*int(abs(minx) + 1)
	maxx += 0.2*int(abs(maxx) + 1)
	miny -= 0.2*int(abs(miny) + 1)
	maxy += 0.2*int(abs(maxy) + 1)
	return (minx, maxx, miny, maxy)

def Fortune(P):
	Q = event_queue(P)
	V = DCEL()
	T = BST()

	for event in Q.keys():
		V.add_face(event.face)

	while Q:
		q = Q.pop()
		q.point.hilight()
		par_plots, sweep = plot_all(T.all_leaves(), V, q.point.y)
		control.sleep()

		if q.is_site_event:
			handle_site_event(q, T, Q, V)
		else:
			handle_circle_event(q, T, Q, V)
			q.point.unplot()
		control.sleep()

		if len(Q) > 0:
			next_y = Q.top().point.y
			line_y = q.point.y
			leaves = T.all_leaves()
			# while not math.isclose(line_y, next_y, rel_tol=4*FORTUNE_PLOT_RATE):
			# FORTUNE_PLOT_RATE = abs(line_y - next_y)/100
			while line_y > next_y:
				control.thaw_sleep()
				if abs(line_y - next_y) > FORTUNE_PLOT_RATE * 10**3:
					line_y -= abs(line_y - next_y)/10
				else:
					line_y -= FORTUNE_PLOT_RATE

				unplot_all(par_plots, V.hedges, sweep)

				par_plots, sweep = plot_all(leaves, V, line_y)
				control.thaw_update ()
				control.update ()
				control.freeze_update ()

		unplot_all(par_plots, V.hedges, sweep)

		q.point.unhilight()
		control.update()

	vertices = [v.p for v in V.vertices]
	borders = find_borders(P + vertices)
	finalize_voronoi(V, T, borders)

	for h in V.hedges:
		h.segment.plot()
	return V

def finalize_voronoi(V, T, borders):
	left = []
	right = []
	bottom = []
	top = []
	for n in T.all_nodes():
		hedge = n.hedge
		bissect_line = bissect_line_function(n)
		if hedge.origin.x() < hedge.dest.x():
			point_x = -FORTUNE_INF
		else:
			point_x = FORTUNE_INF
		point = Point(point_x, bissect_line(point_x))
		hedge.update_origin(point)
		clip_hedge(hedge, borders)

		if math.isclose(hedge.origin.x(), borders[0]):
			left += [hedge.origin]
		elif math.isclose(hedge.origin.x(), borders[1]):
			right += [hedge.origin]
		elif math.isclose(hedge.origin.y(), borders[2]):
			bottom += [hedge.origin]
		elif math.isclose(hedge.origin.y(), borders[3]):
			top += [hedge.origin]

		point.plot('red', 5)

	get_x = lambda point : point.x()
	get_y = lambda point : point.y()
	left.sort(key=get_y, reverse=True)
	right.sort(key=get_y)
	bottom.sort(key=get_x)
	top.sort(key=get_x, reverse=True)

	v_lb = V.add_vertex(Point(borders[0], borders[2]))
	v_rb = V.add_vertex(Point(borders[1], borders[2]))
	v_lt = V.add_vertex(Point(borders[0], borders[3]))
	v_rt = V.add_vertex(Point(borders[1], borders[3]))

	outer_face = Face()
	V.add_face(outer_face)
	h_b1, h_b2 = connect_edge(V, v_lb, v_rb, bottom, outer_face)
	h_r1, h_r2 = connect_edge(V, v_rb, v_rt, right, outer_face)
	h_t1, h_t2 = connect_edge(V, v_rt, v_lt, top, outer_face)
	h_l1, h_l2 = connect_edge(V, v_lt, v_lb, left, outer_face)

	connect_corner(h_b1, h_b2, h_l2, h_r1)
	connect_corner(h_r1, h_r2, h_b2, h_t1)
	connect_corner(h_t1, h_t2, h_r2, h_l1)
	connect_corner(h_l1, h_l2, h_t2, h_b1)

def connect_corner(corner_hedge1, corner_hedge2, prev_hedge, next_hedge):
	corner_hedge1.twin.next_hedge = prev_hedge.twin
	if prev_hedge.twin.face is not None:
		corner_hedge1.add_face(prev_hedge.twin.face)
	corner_hedge2.next_hedge = next_hedge

def connect_edge(V, corner1, corner2, vertices, outer_face):
	previous_v = corner1
	previous_v_hedge = previous_v.hedge
	previous_hedge = None

	for v in vertices:
		h1 = Hedge(previous_v, v)
		V.add_hedge(h1)
		if previous_hedge is not None:
			previous_hedge.next_hedge = h1
		h1.add_face(outer_face)

		inner_hedge = v.hedge
		h2 = Hedge(v, previous_v)
		V.add_hedge(h2)
		if previous_hedge is not None:
			h2.next_hedge = previous_v_hedge
			h2.add_face(previous_v_hedge.face)
		inner_hedge.twin.next_hedge = h2
		h1.add_twin(h2)

		previous_hedge = h1
		previous_v = v
		previous_v_hedge = inner_hedge

	h1 = Hedge(previous_v, corner2)
	V.add_hedge(h1)
	if previous_hedge is not None:
		previous_hedge.next_hedge = h1
	h1.add_face(outer_face)

	h2 = Hedge(corner2, previous_v)
	V.add_hedge(h2)
	if previous_hedge is not None:
		h2.next_hedge = previous_v_hedge
		h2.add_face(previous_v.hedge.face)
	h1.add_twin(h2)

	return corner1.hedge, h1

def clip_hedge(hedge, borders):
	t0 = 0
	t1 = 1
	x_delta = hedge.dest.x() - hedge.origin.x()
	y_delta = hedge.dest.y() - hedge.origin.y()

	p_values = [x_delta, y_delta]
	for i in range(4):
		if i == 0:
			p = -x_delta
			q = -(borders[i] - hedge.origin.x())
		elif i == 1:
			p = x_delta
			q = (borders[i] - hedge.origin.x())
		elif i == 2:
			p = -y_delta
			q = -(borders[i] - hedge.origin.y())
		elif i == 3:
			p = y_delta
			q = borders[i] - hedge.origin.y()

		r = q/p

		if p == 0 and q < 0:
			return False

		if p < 0:
			if r > t1:
				return False
			elif r > t0:
				t0 = r
		elif p > 0:
			if r < t0:
				return False
			elif r < t1:
				t1 = r

	x0_clip = hedge.origin.x() + t0 * x_delta
	y0_clip = hedge.origin.y() + t0 * y_delta
	x1_clip = hedge.origin.x() + t1 * x_delta
	y1_clip = hedge.origin.y() + t1 * y_delta

	hedge.update_origin(Point(x0_clip, y0_clip))
	hedge.update_dest(Point(x1_clip, y1_clip))

	return True

def handle_site_event(q, T, Q, V):
	if T.is_empty():
		T.insert(q)
	else:
		f = T.search(q.point)
		if f.event is not None:
			f.event.point.unplot()
			Q.updateitem(f.event, Point(math.inf, math.inf))
			Q.pop()
			f.event = None

		u, f, v = T.split_and_insert(f, q)

		bissect_line = bissect_line_function(u)
		v_1 = V.add_vertex(Point(None, None))
		v_2 = V.add_vertex(Point(None, None))

		h_12 = Hedge(v_1, v_2)
		V.add_hedge(h_12)
		u.hedge = h_12
		h_12.add_face(u.p_j.face)

		h_21 = Hedge(v_2, v_1)
		V.add_hedge(h_21)
		v.hedge = h_21
		h_21.add_face(v.p_j.face)

		h_12.add_twin(h_21)

		update_events(Q, T, f, f, q.point)


def handle_circle_event(q, T, Q, V):
	f = q.leaf
	pred, succ, new_node = T.remove(f, Q)

	left_leaf = new_node.p_i
	right_leaf = new_node.p_j

	update_events(Q, T, left_leaf, right_leaf, q.point)

	u = V.add_vertex(q.center)

	update_hedge(pred, q, u)
	update_hedge(succ, q, u)
	succ.hedge.twin.next_hedge = pred.hedge
	q.center.plot('red', 5)

	mid1 = mid_point(left_leaf.point, right_leaf.point)
	slope1 = perp_slope(get_line(left_leaf.point, right_leaf.point))

	bissect_line = lambda y : (y - mid1.y)/slope1 + mid1.x
	v = V.add_vertex(Point(bissect_line(-FORTUNE_INF), -FORTUNE_INF))
	h_vu = Hedge(v, u)
	V.add_hedge(h_vu)
	new_node.hedge = h_vu
	h_vu.add_face(new_node.p_j.face)

	h_uv = Hedge(u, v)
	V.add_hedge(h_uv)
	h_uv.add_face(new_node.p_i.face)

	h_uv.add_twin(h_vu)

	pred.hedge.twin.next_hedge = h_uv
	h_vu.next_hedge = succ.hedge

def update_hedge(node, event, vertex):
	node.hedge.update_origin(vertex)

	x_breakpoints = get_x_breakpoints(node, event.point.y)
	bissec = bissect_line_function(node)

	if math.isclose(x_breakpoints[0], event.center.x, rel_tol=FORTUNE_EPS):
		point = Point(FORTUNE_INF, bissec(FORTUNE_INF))
	else:
		point = Point(-FORTUNE_INF, bissec(-FORTUNE_INF))

	if abs(node.hedge.dest.p.x) == FORTUNE_INF:
		node.hedge.update_dest(point)

def is_there_left_triple(leaf):
	return leaf.pred is not None and leaf.pred.p_i.pred is not None

def is_there_right_triple(leaf):
	return leaf.succ is not None and leaf.succ.p_j.succ is not None

def update_events(Q, T, left_leaf, right_leaf, q):
	if is_there_left_triple(right_leaf):
		node1 = right_leaf.pred
		leaf2 = node1.p_i
		node2 = leaf2.pred
		leaf3 = node2.p_i
		if leaf2.event is None:
			add_circle_event(right_leaf, leaf2, leaf3, node1, node2, q, Q)

	if is_there_right_triple(left_leaf):
		node1 = left_leaf.succ
		leaf2 = node1.p_j
		node2 = leaf2.succ
		leaf3 = node2.p_j
		if leaf2.event is None:
			add_circle_event(left_leaf, leaf2, leaf3, node1, node2, q, Q)

def is_valid_event(center, radius, q):
	is_under_sweep = center.y - radius < q.y - FORTUNE_EPS
	is_on_sweep = math.isclose(center.y - radius, q.y)
	is_on_the_right = center.x - radius > q.x + FORTUNE_EPS

	return is_under_sweep or (is_on_sweep and is_on_the_right)

def is_divergent(node, center):
	if node.p_i.point.y < node.p_j.point.y:
		return node.p_i.point.x > center.x
	else:
		return node.p_j.point.x < center.x

def add_circle_event(leaf1, leaf2, leaf3, node1, node2, q, Q):
	p1, p2, p3 = leaf1.point, leaf2.point, leaf3.point
	p2.hilight('yellow')
	p3.hilight('yellow')

	center = circumcenter(p1, p2, p3)
	radius = distance(center, p1)
	circle = control.plot_circle(center.x, center.y, 'blue', radius)

	is_convergent = not(is_divergent(node1, center) and is_divergent(node2, center))
	if is_valid_event(center, radius, q) and is_convergent:
		point = Point(center.x, center.y - radius)
		leaf2.event = Event(point, False, leaf2, center)
		Q.additem(leaf2.event, point)
		point.plot(color='cyan')

	control.sleep()
	control.plot_delete(circle)
	p2.unhilight()
	p3.unhilight()
