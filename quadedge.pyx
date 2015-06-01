# Quadedge implementation based on Lischinski's code published in
# Graphics Gems IV: https://github.com/erich666/GraphicsGems/tree/master/gemsiv/delaunay

# Use real division everywhere
from __future__ import division

import numpy as np
cimport numpy as np
from cpython cimport array as c_array
from array import array
import sys
from numbers import Number


np.import_array()

DTYPE = np.int
ctypedef np.int_t DTYPE_t

from cython.operator cimport postincrement as postinc
import math

from libc.math cimport sqrt, fabs

cdef float eps = 1e-6
cdef float_min = -sys.float_info.max

cdef class Edge:
    def __cinit__(self, origin = None):
        self.origin = origin

    def __str__(self):
        return "(" + str(self.origin.x) + "," + str(self.origin.y) + \
            ") -- (" + str(self.destination.x) + "," + str(self.destination.y) + ")"

    property destination:
        def __get__(self): return self.sym.origin
        def __set__(self, dest): self.sym.origin = dest

    property rot:
        def __get__(self): return self._rot

    property sym:
        def __get__(self): return self._rot._rot

    property invRot:
        def __get__(self): return self._rot._rot._rot

    property oNext:
        def __get__(self): return self.next

    property oPrev:
        def __get__(self): return self.rot.next.rot

    property dNext:
        def __get__(self): return self.sym.next.sym

    property dPrev:
        def __get__(self): return self.invRot.next.invRot

    property lNext:
        def __get__(self): return self.invRot.next.rot

    property lPrev:
        def __get__(self): return self.next.sym

    property rNext:
        def __get__(self): return self.rot.next.invRot

    property rPrev:
        def __get__(self): return self.sym.next
    
    property length:
        def __get__(self):
            vec = self.origin - self.destination
            return np.sqrt(vec.x**2 + vec.y**2)
    
    property is_boundary:
        def __get__(self):
            return self.oPrev.destination.rightOf(self)

    cpdef asLineSegment(self):
        return [self.origin.pos, self.destination.pos]
    
    

cdef class QuadEdge:
    cdef public list edges

    def __cinit__(self, Vertex origin, Vertex destination):
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

    property base:
        def __get__(self): return self.edges[0]

cpdef Edge splice(Edge a, Edge b):
    cdef Edge alpha, beta, t1, t2, t3, t4
    alpha = a.oNext.rot
    beta = b.oNext.rot

    t1 = b.oNext
    t2 = a.oNext
    t3 = beta.oNext
    t4 = alpha.oNext

    a.next = t1
    b.next = t2
    alpha.next = t3
    beta.next = t4

cpdef Edge connect(Edge a, Edge b):
    e = QuadEdge(a.destination, b.origin).base
    splice(e, a.lNext)
    splice(e.sym, b)
    return e

cpdef Edge makeEdge(Vertex origin, Vertex destination):
    e = QuadEdge(origin, destination).base
    return e

cpdef deleteEdge(Edge e):
    splice(e, e.oPrev)
    splice(e.sym, e.sym.oPrev)
    del e

cpdef swap(Edge e):
    cdef Edge a, b
    a = e.oPrev
    b = e.sym.oPrev
    splice(e, a)
    splice(e.sym, b)
    splice(e, a.lNext)
    splice(e.sym, b.lNext)
    e.origin = a.destination
    e.destination = b.destination



cpdef makeTriangle(Vertex v0, Vertex v1, Vertex v2):
    e0 = makeEdge(v0, v1)
    e1 = makeEdge(v1, v2)
    e2 = makeEdge(v2, v0)
    splice(e0.sym, e1)
    splice(e1.sym, e2)
    splice(e2.sym, e0)
    return e0

cdef class Vertex:

    def __cinit__(self, long x, long y, float z = 0):
        self.x = x
        self.y = y
        self.z = z
    
    property pos:
        def __get__(self):
            return (self.x, self.y, self.z)
        def __set__(self, pos):
            self.x = pos[0]
            self.y = pos[1]
            self.z = pos[2]
    
    cpdef inTriangle(self, Vertex v0, Vertex v1, Vertex v2):
        return triangleArea(v0, v1, self) >= 0 and \
               triangleArea(v1, v2, self) >= 0 and \
               triangleArea(v2, v0, self) >= 0

    cpdef inCircle(self, Vertex v0, Vertex v1, Vertex v2):
        return (  v0.x ** 2 +   v0.y ** 2) * triangleArea(v1, v2, self) - \
               (  v1.x ** 2 +   v1.y ** 2) * triangleArea(v0, v2, self) + \
               (  v2.x ** 2 +   v2.y ** 2) * triangleArea(v0, v1, self) - \
               (self.x ** 2 + self.y ** 2) * triangleArea(v0, v1,   v2) > eps

    cpdef leftOf(self, Edge e):
        return ccw(self, e.origin, e.destination)

    cpdef rightOf(self, Edge e):
        return ccw(self, e.destination, e.origin)

    cpdef onEdge(self, Edge e):
        cdef float t1, t2, t3
        t1 = (self - e.origin).norm
        t2 = (self - e.destination).norm
        if (t1 < eps or t2 < eps):
            return True
        t3 = (e.origin - e.destination).norm
        if t1 > t3 or t2 > t3:
            return False
        l = Line(e.origin, e.destination)
        return fabs(l.evaluate(self)) < eps

    property norm:
        def __get__(self): return sqrt(self.x ** 2 + self.y ** 2)

    def __add__(self, other):
        return Vertex(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vertex(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, other):
        if isinstance(other, Vertex):
            return self.x * other.x + self.y * other.y
        elif isinstance(other, Number):
            return Vertex(self.x * other, self.y * other, self.z * other)

    def __div__(self, other):
        if isinstance(other, Number):
            return Vertex(self.x / other, self.y / other, self.z / other)    
    
    def __richcmp__(self, other, operation):
        if operation == 2:
            return self.x == other.x and self.y == other.y and self.z == other.z
        elif operation == 3:
            return not(self.x == other.x and self.y == other.y and self.z == other.z)
        else:
            return False
    
    def __str__(self):
        return "({},{},{})".format(*self.pos)

cdef class Triangle:       
    def __cinit__(self, Edge e, anchor = True, ID = -1):
        self.vertices = [e.origin, e.destination, e.lPrev.origin]
        self.area = triangleArea(*self.vertices)        
        self.ID = ID
        self.candidate = Vertex(-1, -1, 0)
        self.candidate_error = float_min
        
        if anchor:
            self.anchor = e
            self.reshape()
        self.children = []
        
        self.calculate_plane_equation()
    
    cpdef reshape(self):
        self.anchor.triangle = self
        self.anchor.lNext.triangle = self
        self.anchor.lPrev.triangle = self
    
    cpdef calculate_plane_equation(self):
        u = self.vertices[1] - self.vertices[0]
        v = self.vertices[2] - self.vertices[0]
        
        den = float(u.x * v.y - u.y * v.x)
        
        self.a = (u.z * v.y - u.y * v.z) / den
        self.b = (u.x * v.z - u.z * v.x) / den
        self.c = self.vertices[0].z - \
                 self.a * self.vertices[0].x - \
                 self.b * self.vertices[0].y
    
    cpdef interpolate(self, int x, int y):
        return self.a * x + self.b * y + self.c

  
    def __str__(self):
        return "{} -- {} -- {}".format(self.vertices[0],
                                       self.vertices[1],
                                       self.vertices[2])
  

cpdef triangleArea(Vertex v0, Vertex v1, Vertex v2):
    return (v1.x - v0.x) * (v2.y - v0.y) - (v1.y - v0.y) * (v2.x - v0.x);

cpdef ccw(Vertex v0, Vertex v1, Vertex v2):
    return triangleArea(v0, v1, v2) > 0

cpdef list edgeRing(Edge e):
    coordList = []
    startEdge = e
    coordList.append(tuple(e.origin.pos))
    coordList.append(tuple(e.destination.pos))
    e = e.lNext
    while e != startEdge:
        coordList.append(tuple(e.destination.pos))
        e = e.lNext
    return coordList



cdef class Line:
    cdef public float a, b, c

    def __cinit__(self, Vertex v0, Vertex v1):
        cdef Vertex t = v1 - v0
        cdef float l = t.norm
        assert(l != 0)
        self.a =  t.y / l
        self.b = -t.x / l
        self.c = -(self.a * v0.x + self.b * v0.y)

    cpdef evaluate(self, Vertex v):
        return self.a * v.x + self.b * v.y + self.c

    cpdef classify(self, Vertex v):
        d = self.evaluate(v)

        if d < -eps:
            return -1
        elif d > eps:
            return 1
        else:
            return 0

    cpdef Vertex intersect(self, Line l):
        den = self.a * l.b - self.b * l.a
        assert(den!=0)
        return Vertex([(self.b * l.c - self.c * l.b) / den,
                       (self.c * l.a - self.a * l.c) / den])