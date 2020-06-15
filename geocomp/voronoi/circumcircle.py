import math
from geocomp.voronoi.point import Point
from geocomp.common import control
from geocomp.common.guiprim import *
from geocomp.common.segment import Segment

def mid_point(p1, p2):
    """Encontra o ponto médio entre dois pontos"""
    return Point((p1.x + p2.x)/2, (p1.y + p2.y)/2)

def get_line(p1, p2):
    """Encontra a reta que corta dois pontos"""
    if p2.x == p1.x:
        slope = math.inf
    else:
        slope = (p2.y - p1.y)/(p2.x - p1.x)
    y_int = p1.y - slope * p1.x
    return (slope, y_int)

def perp_slope(line):
    """Encontra o coeficiente angula da reta perpedicular"""
    return -1 / line[0]

def get_line_from_slope(slope, point):
    """Encontra a reta dado um ponto e seu coeficiente angular"""
    return (slope, point.y - slope*point.x)

def intersection(line1, line2):
    """Encontra o ponto de interseção de duas retas"""
    x = (line2[1] - line1[1])/(line1[0] - line2[0])
    y = line1[0] * x + line1[1]
    return Point(x, y)

def distance(p1, p2):
    """Retorna a distância entre dois pontos"""
    return math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)

def bissect_line_function(node):
    """Retorna a função da linha bissetora representada pelo nó interno"""
    mid = mid_point(node.p_i.point, node.p_j.point)
    slope = perp_slope(get_line(node.p_i.point, node.p_j.point))

    return lambda x : slope*x + mid.y - mid.x*slope

def circumcenter(p1, p2, p3):
    """Encontra o cicuncentro de três pontos"""
    mid1 = mid_point(p1, p2)
    mid2 = mid_point(p2, p3)

    line1 = get_line(p1, p2)
    line2 = get_line(p2, p3)

    perp1 = perp_slope(line1)
    perp2 = perp_slope(line2)

    bissect1 = get_line_from_slope(perp1, mid1)
    bissect2 = get_line_from_slope(perp2, mid2)

    inter = intersection(bissect1, bissect2)
    return inter
