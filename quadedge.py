# Quadedge implementation based on Lischinski's code published in
# Graphics Gems IV: https://github.com/erich666/GraphicsGems/tree/master/gemsiv/delaunay

import numpy as np
from array import array
import sys
from numbers import Number

import math


eps = 1e-6
float_min = -sys.float_info.max


class Edge:
    def __init__(self, origin=None):
        self.origin = origin
        self.rot = self
        self.next = self

    def __str__(self):
        return "(" + str(self.origin.x) + "," + str(self.origin.y) + \
            ") -- (" + str(self.destination.x) + "," + str(self.destination.y) + ")"

    @property
    def destination(self): return self.sym.origin

    @destination.setter
    def destination(self, dest): self.sym.origin = dest

    @property
    def sym(self): return self.rot.rot

    @property
    def inv_rot(self): return self.rot.rot.rot

    @property
    def o_next(self): return self.next

    @property
    def o_prev(self): return self.rot.next.rot

    @property
    def d_next(self): return self.sym.next.sym

    @property
    def d_prev(self): return self.inv_rot.next.inv_rot

    @property
    def l_next(self): return self.inv_rot.next.rot

    @property
    def l_prev(self): return self.next.sym

    @property
    def r_next(self): return self.rot.next.inv_rot

    @property
    def r_prev(self): return self.sym.next
    
    @property
    def length(self):
        vec = self.origin - self.destination
        return np.sqrt(vec.x**2 + vec.y**2)
    
    @property
    def is_boundary(self):
        return self.o_prev.destination.right_of(self)

    def as_line_segment(self):
        return [self.origin.pos, self.destination.pos]
    

class QuadEdge:
    def __init__(self, origin=None, destination=None):
        self.edges = [
            Edge(origin),
            Edge(),
            Edge(destination),
            Edge()
        ]

        self.edges[0].rot = self.edges[1]
        self.edges[1].rot = self.edges[2]
        self.edges[2].rot = self.edges[3]
        self.edges[3].rot = self.edges[0]

        self.edges[0].next = self.edges[0]
        self.edges[1].next = self.edges[3]
        self.edges[2].next = self.edges[2]
        self.edges[3].next = self.edges[1]

    @property
    def base(self): return self.edges[0]


def splice(a, b):
    alpha = a.o_next.rot
    beta = b.o_next.rot

    t1 = b.o_next
    t2 = a.o_next
    t3 = beta.o_next
    t4 = alpha.o_next

    a.next = t1
    b.next = t2
    alpha.next = t3
    beta.next = t4


def connect(a, b):
    e = QuadEdge(a.destination, b.origin).base
    splice(e, a.l_next)
    splice(e.sym, b)
    return e


def make_edge(origin, destination):
    e = QuadEdge(origin, destination).base
    return e


def delete_edge(e):
    splice(e, e.o_prev)
    splice(e.sym, e.sym.o_prev)
    del e


def swap(e):
    a = e.o_prev
    b = e.sym.o_prev
    splice(e, a)
    splice(e.sym, b)
    splice(e, a.l_next)
    splice(e.sym, b.l_next)
    e.origin = a.destination
    e.destination = b.destination


def make_triangle(v0, v1, v2):
    e0 = make_edge(v0, v1)
    e1 = make_edge(v1, v2)
    e2 = make_edge(v2, v0)
    splice(e0.sym, e1)
    splice(e1.sym, e2)
    splice(e2.sym, e0)
    return e0


class Vertex:
    def __init__(self, x, y, z=0):
        self.x = x
        self.y = y
        self.z = z

    @property
    def pos(self): return self.x, self.y, self.z

    @pos.setter
    def pos(self, pos):
        self.x = pos[0]
        self.y = pos[1]
        self.z = pos[2]

    def in_triangle(self, v0, v1, v2):
        return triangle_area(v0, v1, self) >= 0 and \
               triangle_area(v1, v2, self) >= 0 and \
               triangle_area(v2, v0, self) >= 0

    def in_circle(self, v0, v1, v2):
        return (  v0.x ** 2 +   v0.y ** 2) * triangle_area(v1, v2, self) - \
               (  v1.x ** 2 +   v1.y ** 2) * triangle_area(v0, v2, self) + \
               (  v2.x ** 2 +   v2.y ** 2) * triangle_area(v0, v1, self) - \
               (self.x ** 2 + self.y ** 2) * triangle_area(v0, v1,   v2) > eps

    def left_of(self, e):
        return ccw(self, e.origin, e.destination)

    def right_of(self, e):
        return ccw(self, e.destination, e.origin)

    def on_edge(self, e):
        t1 = (self - e.origin).norm
        t2 = (self - e.destination).norm
        if t1 < eps or t2 < eps:
            return True
        t3 = (e.origin - e.destination).norm
        if t1 > t3 or t2 > t3:
            return False
        l = Line(e.origin, e.destination)
        return math.fabs(l.evaluate(self)) < eps

    @property
    def norm(self): return math.sqrt(self.x ** 2 + self.y ** 2)

    def __add__(self, other):
        if isinstance(other, Vertex):
            return Vertex(self.x + other.x, self.y + other.y, self.z + other.z)
        else:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Vertex):
            return Vertex(self.x - other.x, self.y - other.y, self.z - other.z)
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Vertex):
            return self.x * other.x + self.y * other.y
        elif isinstance(other, Number):
            return Vertex(self.x * other, self.y * other, self.z * other)
        else:
            return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, Number):
            return Vertex(self.x / other, self.y / other, self.z / other)
        else:
            return NotImplemented

    def __eq__(self, other):
        if isinstance(other, Vertex):
            return self.x == other.x and self.y == other.y and self.z == other.z
        else:
            return NotImplemented

    def __ne__(self, other):
        if isinstance(other, Vertex):
            return not(self.x == other.x and self.y == other.y and self.z == other.z)
        else:
            return NotImplemented

    def __str__(self):
        return "({},{},{})".format(*self.pos)


class Triangle:
    def __init__(self, e, anchor=True, id=-1):
        self.vertices = [e.origin, e.destination, e.l_prev.origin]
        self.area = triangle_area(*self.vertices)
        self.id = id
        self.candidate = Vertex(-1, -1, 0)
        self.candidate_error = float_min
        self.a = self.b = self.c = None

        if anchor:
            self.anchor = e
            self.reshape()
        self.children = []

        self.calculate_plane_equation()

    def reshape(self):
        self.anchor.triangle = self
        self.anchor.l_next.triangle = self
        self.anchor.l_prev.triangle = self

    def calculate_plane_equation(self):
        u = self.vertices[1] - self.vertices[0]
        v = self.vertices[2] - self.vertices[0]

        den = float(u.x * v.y - u.y * v.x)

        self.a = (u.z * v.y - u.y * v.z) / den
        self.b = (u.x * v.z - u.z * v.x) / den
        self.c = self.vertices[0].z - \
                 self.a * self.vertices[0].x - \
                 self.b * self.vertices[0].y

    def interpolate(self, x, y):
        return self.a * x + self.b * y + self.c

    def __str__(self):
        return "{} -- {} -- {}".format(self.vertices[0],
                                       self.vertices[1],
                                       self.vertices[2])


def triangle_area(v0, v1, v2):
    return (v1.x - v0.x) * (v2.y - v0.y) - (v1.y - v0.y) * (v2.x - v0.x);


def ccw(v0, v1, v2):
    return triangle_area(v0, v1, v2) > 0


def edge_ring(e):
    coord_list = []
    start_edge = e
    coord_list.append(tuple(e.origin.pos))
    coord_list.append(tuple(e.destination.pos))
    e = e.l_next
    while e != start_edge:
        coord_list.append(tuple(e.destination.pos))
        e = e.l_next
    return coord_list


class Line:
    def __init__(self, v0, v1):
        t = v1 - v0
        l = t.norm
        assert(l != 0)
        self.a =  t.y / l
        self.b = -t.x / l
        self.c = -(self.a * v0.x + self.b * v0.y)

    def evaluate(self, v):
        return self.a * v.x + self.b * v.y + self.c

    def classify(self, v):
        d = self.evaluate(v)

        if d < -eps:
            return -1
        elif d > eps:
            return 1
        else:
            return 0

    def intersect(self, l):
        den = self.a * l.b - self.b * l.a
        assert(den != 0)
        return Vertex((self.b * l.c - self.c * l.b) / den,
                      (self.c * l.a - self.a * l.c) / den)
