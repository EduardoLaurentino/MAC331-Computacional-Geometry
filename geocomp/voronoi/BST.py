import math
from geocomp.voronoi.point import Point
from geocomp.common import control

class Node():
    """Implementa um nó interno"""
    def __init__(self, left_leaf, right_leaf):
        self.p_i = left_leaf
        self.p_j = right_leaf
        self.right = self.left = self.hedge = self.father = None

    def __repr__(self):
        return f'<Node: ({int(self.p_i.point.x)}, {int(self.p_i.point.y)}), ({int(self.p_j.point.x)}, {int(self.p_j.point.y)})>'

class Leaf():
    """Implementa uma folha"""
    def __init__(self, point, pred=None, succ=None, face=None):
        self.point = point
        self.event = None
        self.pred = pred
        self.succ = succ
        self.face = face

    def __repr__(self):
        return f'<Leaf: ({int(self.point.x)}, {int(self.point.y)})>'

class BST():
    """Implementa uma árvore de busca balanceada"""

    def __init__(self):
        self.root = None

    def is_empty(self):
        return self.root is None

    def insert(self, event):
        """Insere um arco em uma árvore vazia"""
        self.root = Leaf(event.point, face=event.face)

    def search(self, point):
        """Busca a folha do arco acima do ponto"""
        def inner_search(node, point):
            if isinstance(node, Leaf):
                return node

            x_breakpoint = get_x_breakpoint(node, point.y)
            if point.x < x_breakpoint:
                return inner_search(node.left, point)
            return inner_search(node.right, point)

        return inner_search(self.root, point)

    def split_and_insert(self, leaf, event):
        """Substitui a folha da árvore pela subárvore com três folhas:

            leaf   =>    new_tree
                        /        \\
                   left_split     new_node
                                 /       \\
                               new_leaf  right_split
        """
        point = event.point

        new_leaf = Leaf(point)
        left_split = Leaf(leaf.point)
        right_split = Leaf(leaf.point)

        new_tree = Node(left_split, new_leaf)
        new_node = Node(new_leaf, right_split)

        left_split.pred, left_split.succ   = leaf.pred, new_tree
        right_split.pred, right_split.succ = new_node, leaf.succ
        new_leaf.pred, new_leaf.succ  = new_tree, new_node

        new_tree.left, new_tree.right = left_split, new_node
        new_node.left, new_node.right = new_leaf, right_split

        new_node.father = new_tree
        if leaf.pred is not None:
            leaf.pred.p_j = left_split
        if leaf.succ is not None:
            leaf.succ.p_i = right_split

        if leaf.pred is not None and leaf.pred.right == leaf:
            new_tree.father = leaf.pred
            leaf.pred.right = new_tree
        elif leaf.succ is not None:
            new_tree.father = leaf.succ
            leaf.succ.left = new_tree
        else:
            self.root = new_tree

        left_split.face = right_split.face = leaf.face
        new_leaf.face = event.face
        return new_tree, new_leaf, new_node

    def remove(self, leaf, Q):
        """Remove uma folha da árvore e devolve os dois nós internos associados
        e seu substituto
                  subst                   new_node
                /      \\                  /    \\
            remov              =>    other_node
           /    \\
         leaf    other_node
        """
        def substitute_node(node, substitute):
            if node == self.root:
                self.root = substitute
            elif node.father.left == node:
                node.father.left = substitute
            else:
                node.father.right = substitute

        def substitute_father(node, substitute):
            if isinstance(node, Node):
                node.father = substitute

        def remove_circle_event(leaf, Q):
            if leaf.event is not None:
                Q.updateitem(leaf.event, Point(math.inf, math.inf))
                Q.pop()
                leaf.event.point.unplot()
                leaf.event = None

        pred, succ = leaf.pred, leaf.succ
        if pred is None:
            substitute_node(succ, succ.right)
            substitute_father(succ.right, succ.father)
            return pred, succ, None
        if succ is None:
            substitute_node(pred, pred.left)
            substitute_father(pred.right, pred.father)
            return pred, succ, None

        new_node = Node(pred.p_i, succ.p_j)
        remov, subst = (pred, succ) if pred.right == leaf else (succ, pred)
        other_node   = remov.right  if remov.left == leaf else remov.left

        pred.p_i.succ = new_node
        succ.p_j.pred = new_node

        new_node.father = subst.father
        new_node.left = subst.left
        new_node.right = subst.right
        substitute_node(subst, new_node)
        substitute_father(subst.left, new_node)
        substitute_father(subst.right, new_node)

        remov.father = remov.father if remov.father != subst else new_node
        substitute_node(remov, other_node)
        substitute_father(other_node, remov.father)

        remove_circle_event(pred.p_i, Q)
        remove_circle_event(succ.p_j, Q)

        return pred, succ, new_node

    def all_nodes(self):
        """Retorna todos os nós internos da árvore em pré-ordem"""
        def inner_all_nodes(node):
            if isinstance(node, Leaf):
                return []

            nodes = [node]
            nodes += inner_all_nodes(node.left)
            nodes += inner_all_nodes(node.right)
            return nodes
        if self.is_empty():
            return []
        return inner_all_nodes(self.root)

    def all_leaves(self):
        """Retorna todas as folhas da árvore em ordem crescente"""
        def inner_all_leaves(node):
            if isinstance(node, Leaf):
                return [node]
            leaves = []
            leaves += inner_all_leaves(node.left)
            leaves += inner_all_leaves(node.right)
            return leaves
        if self.is_empty():
            return []
        return inner_all_leaves(self.root)

    def _str_children(self, node):
        queue = [node]
        string = ''
        count = div = 1
        while len(queue) > 0:
            n = queue.pop(0)
            string += repr(n)
            if isinstance(n, Node):
                queue.append(n.left)
                queue.append(n.right)
            string += '\n'
            count += 1
        return string

    def __str__(self):
        return self._str_children(self.root)

def derivative_parable(p, line_y, x_breakpoint):
    if p.y == line_y:
        return math.inf
    return (x_breakpoint - p.x)/(p.y - line_y)

def get_x_breakpoint(node, line_y):
    x_breakpoints = get_x_breakpoints(node, line_y)
    return choose_x_breakpoint(node, x_breakpoints, line_y)

def get_x_breakpoints(node, line_y):
    """ Calcula as coordenadas x do breakpoint dado a tupla de pontos
    e a posição y da linha de varredura
    """
    i, j = node.p_i.point, node.p_j.point

    a = j.y - i.y
    b = 2 * (j.x*i.y - i.x*j.y + line_y * (i.x - j.x))
    c = (j.y - line_y) * (i.x**2 + i.y**2 - line_y**2)\
        - (j.x**2 + j.y**2 - line_y**2) * (i.y - line_y)

    delta = b**2 - 4*a*c
    if delta < 0:
        raise ValueError('Negative discriminant')
    roots = [(-b + math.sqrt(delta))/(2*a), (-b - math.sqrt(delta))/(2*a)]
    return sorted(roots)

def choose_x_breakpoint(node, x_breakpoints, line_y):
    """ Escolhe a coordenada x do breakpoint referente ao nó dado """
    p_i, p_j = node.p_i.point, node.p_j.point
    d_pi0 = derivative_parable(p_i, line_y, x_breakpoints[0])
    d_pj0 = derivative_parable(p_j, line_y, x_breakpoints[0])

    if d_pi0 < d_pj0:
        return x_breakpoints[1]
    elif d_pi0 > d_pj0:
        return x_breakpoints[0]
    else:
        raise ValueError('Same derivative parabolas')
