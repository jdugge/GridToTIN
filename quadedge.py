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
        self._rot = self

    def __str__(self):
        return "(" + str(self.origin.x) + "," + str(self.origin.y) + \
            ") -- (" + str(self.destination.x) + "," + str(self.destination.y) + ")"

    @property
    def destination(self): return self.sym.origin
    @destination.setter
    def destination(self, dest): self.sym.origin = dest

    @property
    def rot(self): return self._rot

    @property
    def sym(self): return self._rot._rot

    @property
    def inv_rot(self): return self._rot._rot._rot

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
        return self.o_prev.destination.rightOf(self)

    def as_line_segment(self):
        return [self.origin.pos, self.destination.pos]
    

class QuadEdge:
    def __init__(self, origin=None, destination=None):
        self.edges = 4*[None]

        self.edges[0] = Edge(origin)
        self.edges[1] = Edge()
        self.edges[2] = Edge(destination)
        self.edges[3] = Edge()

        self.edges[0]._rot = self.edges[1]
        self.edges[1]._rot = self.edges[2]
        self.edges[2]._rot = self.edges[3]
        self.edges[3]._rot = self.edges[0]

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
#
# cpdef Edge connect(Edge a, Edge b):
#     e = QuadEdge(a.destination, b.origin).base
#     splice(e, a.l_next)
#     splice(e.sym, b)
#     return e
#
# cpdef Edge makeEdge(Vertex origin, Vertex destination):
#     e = QuadEdge(origin, destination).base
#     return e
#
# cpdef deleteEdge(Edge e):
#     splice(e, e.o_prev)
#     splice(e.sym, e.sym.o_prev)
#     del e
#
# cpdef swap(Edge e):
#     cdef Edge a, b
#     a = e.o_prev
#     b = e.sym.o_prev
#     splice(e, a)
#     splice(e.sym, b)
#     splice(e, a.l_next)
#     splice(e.sym, b.l_next)
#     e.origin = a.destination
#     e.destination = b.destination
#
#
#
# cpdef makeTriangle(Vertex v0, Vertex v1, Vertex v2):
#     e0 = makeEdge(v0, v1)
#     e1 = makeEdge(v1, v2)
#     e2 = makeEdge(v2, v0)
#     splice(e0.sym, e1)
#     splice(e1.sym, e2)
#     splice(e2.sym, e0)
#     return e0
#
# cdef class Vertex:
#
#     def __cinit__(self, long x, long y, float z = 0):
#         self.x = x
#         self.y = y
#         self.z = z
#
#     property pos:
#         def __get__(self):
#             return (self.x, self.y, self.z)
#         def __set__(self, pos):
#             self.x = pos[0]
#             self.y = pos[1]
#             self.z = pos[2]
#
#     cpdef inTriangle(self, Vertex v0, Vertex v1, Vertex v2):
#         return triangleArea(v0, v1, self) >= 0 and \
#                triangleArea(v1, v2, self) >= 0 and \
#                triangleArea(v2, v0, self) >= 0
#
#     cpdef inCircle(self, Vertex v0, Vertex v1, Vertex v2):
#         return (  v0.x ** 2 +   v0.y ** 2) * triangleArea(v1, v2, self) - \
#                (  v1.x ** 2 +   v1.y ** 2) * triangleArea(v0, v2, self) + \
#                (  v2.x ** 2 +   v2.y ** 2) * triangleArea(v0, v1, self) - \
#                (self.x ** 2 + self.y ** 2) * triangleArea(v0, v1,   v2) > eps
#
#     cpdef leftOf(self, Edge e):
#         return ccw(self, e.origin, e.destination)
#
#     cpdef rightOf(self, Edge e):
#         return ccw(self, e.destination, e.origin)
#
#     cpdef onEdge(self, Edge e):
#         cdef float t1, t2, t3
#         t1 = (self - e.origin).norm
#         t2 = (self - e.destination).norm
#         if (t1 < eps or t2 < eps):
#             return True
#         t3 = (e.origin - e.destination).norm
#         if t1 > t3 or t2 > t3:
#             return False
#         l = Line(e.origin, e.destination)
#         return fabs(l.evaluate(self)) < eps
#
#     property norm:
#         def __get__(self): return sqrt(self.x ** 2 + self.y ** 2)
#
#     def __add__(self, other):
#         return Vertex(self.x + other.x, self.y + other.y, self.z + other.z)
#
#     def __sub__(self, other):
#         return Vertex(self.x - other.x, self.y - other.y, self.z - other.z)
#
#     def __mul__(self, other):
#         if isinstance(other, Vertex):
#             return self.x * other.x + self.y * other.y
#         elif isinstance(other, Number):
#             return Vertex(self.x * other, self.y * other, self.z * other)
#
#     def __div__(self, other):
#         if isinstance(other, Number):
#             return Vertex(self.x / other, self.y / other, self.z / other)
#
#     def __richcmp__(self, other, operation):
#         if operation == 2:
#             return self.x == other.x and self.y == other.y and self.z == other.z
#         elif operation == 3:
#             return not(self.x == other.x and self.y == other.y and self.z == other.z)
#         else:
#             return False
#
#     def __str__(self):
#         return "({},{},{})".format(*self.pos)
#
# cdef class Triangle:
#     def __cinit__(self, Edge e, anchor = True, ID = -1):
#         self.vertices = [e.origin, e.destination, e.l_prev.origin]
#         self.area = triangleArea(*self.vertices)
#         self.ID = ID
#         self.candidate = Vertex(-1, -1, 0)
#         self.candidate_error = float_min
#
#         if anchor:
#             self.anchor = e
#             self.reshape()
#         self.children = []
#
#         self.calculate_plane_equation()
#
#     cpdef reshape(self):
#         self.anchor.triangle = self
#         self.anchor.l_next.triangle = self
#         self.anchor.l_prev.triangle = self
#
#     cpdef calculate_plane_equation(self):
#         u = self.vertices[1] - self.vertices[0]
#         v = self.vertices[2] - self.vertices[0]
#
#         den = float(u.x * v.y - u.y * v.x)
#
#         self.a = (u.z * v.y - u.y * v.z) / den
#         self.b = (u.x * v.z - u.z * v.x) / den
#         self.c = self.vertices[0].z - \
#                  self.a * self.vertices[0].x - \
#                  self.b * self.vertices[0].y
#
#     cpdef interpolate(self, int x, int y):
#         return self.a * x + self.b * y + self.c
#
#
#     def __str__(self):
#         return "{} -- {} -- {}".format(self.vertices[0],
#                                        self.vertices[1],
#                                        self.vertices[2])
#
#
# cpdef triangleArea(Vertex v0, Vertex v1, Vertex v2):
#     return (v1.x - v0.x) * (v2.y - v0.y) - (v1.y - v0.y) * (v2.x - v0.x);
#
# cpdef ccw(Vertex v0, Vertex v1, Vertex v2):
#     return triangleArea(v0, v1, v2) > 0
#
# cpdef list edgeRing(Edge e):
#     coordList = []
#     startEdge = e
#     coordList.append(tuple(e.origin.pos))
#     coordList.append(tuple(e.destination.pos))
#     e = e.l_next
#     while e != startEdge:
#         coordList.append(tuple(e.destination.pos))
#         e = e.l_next
#     return coordList
#
#
#
# cdef class Line:
#     cdef public float a, b, c
#
#     def __cinit__(self, Vertex v0, Vertex v1):
#         cdef Vertex t = v1 - v0
#         cdef float l = t.norm
#         assert(l != 0)
#         self.a =  t.y / l
#         self.b = -t.x / l
#         self.c = -(self.a * v0.x + self.b * v0.y)
#
#     cpdef evaluate(self, Vertex v):
#         return self.a * v.x + self.b * v.y + self.c
#
#     cpdef classify(self, Vertex v):
#         d = self.evaluate(v)
#
#         if d < -eps:
#             return -1
#         elif d > eps:
#             return 1
#         else:
#             return 0
#
#     cpdef Vertex intersect(self, Line l):
#         den = self.a * l.b - self.b * l.a
#         assert(den!=0)
#         return Vertex([(self.b * l.c - self.c * l.b) / den,
#                        (self.c * l.a - self.a * l.c) / den])