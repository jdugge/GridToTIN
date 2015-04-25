import numpy as np
cimport numpy as np
from cython.operator cimport postincrement as postinc

from libc.math cimport sqrt, fabs

cdef float eps = 1e-6

cdef int nextID = 0
cpdef int getNextID():
    global nextID
    return postinc(nextID)

cdef class Vertex:
    cdef public long x, y
    cdef public float z
    cdef public Edge parent

    def __cinit__(self, long x, long y, float z = 0):
        self.x = x
        self.y = y
        self.z = z
    
    property pos:
        def __get__(self):
            return (self.x, self.y, self.z)

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

    def __richcmp__(self, other, operation):
        if operation == 2:
            return self.x == other.x and self.y == other.y and self.z == other.z
        elif operation == 3:
            return not(self.x == other.x and self.y == other.y and self.z == other.z)
        else:
            return False

cdef class Face:
    cdef public int ID
    cdef public Edge edge

cdef class Edge:
    cdef public int ID
    cdef public unsigned int index
    cdef public int flag
    cdef public Edge next
    cdef public Edge _rot
    cdef public Vertex origin

    def __cinit__(self, origin = None, ID = None):
        self.origin = origin
        self.ID = ID if ID is not None else getNextID()

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

    cpdef asLineSegment(self):
        return [self.origin.pos, self.destination.pos]

def addEdge(e, l, flag):
    if e.flag != flag:
        l.append(e)
        e.flag = flag
        addEdge(e.oNext, l, flag)
        addEdge(e.oPrev, l, flag)
        addEdge(e.dNext, l, flag)
        addEdge(e.dPrev, l, flag)
        #edgeList = []
        #addEdge(q1, edgeList, 1)

cdef class QuadEdge:
    cdef public list edges

    def __cinit__(self, origin, destination):
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

cpdef triangleArea(Vertex v0, Vertex v1, Vertex v2):
    return (v1.x - v0.x) * (v2.y - v0.y) - (v1.y - v0.y) * (v2.x - v0.x);

cpdef ccw(Vertex v0, Vertex v1, Vertex v2):
    return triangleArea(v0, v1, v2) > 0

cpdef locate(Vertex v, Edge e):
    while (True):
        if (v == e.origin or v == e.destination):
            return e
        elif v.rightOf(e):
            e = e.sym
        elif not (v.rightOf(e.oNext)):
            e = e.oNext
        elif not (v.rightOf(e.dPrev)):
            e = e.dPrev
        else:
            return e

cpdef insertSite(Vertex v, Edge e):
    e = locate(v, e)
    if v == e.origin or v == e.destination:
        pass
    elif v.onEdge(e):
        e = e.oPrev
        deleteEdge(e.oNext)
    base = makeEdge(e.origin, v)
    splice(base, e)
    startingEdge = base

    base = connect(e, base.sym)
    e = base.oPrev
    while e.lNext != startingEdge:
        base = connect(e, base.sym)
        e = base.oPrev

    while True:
        t = e.oPrev
        if t.destination.rightOf(e) and v.inCircle(e.origin,
                                                   t.destination,
                                                   e.destination):
            swap(e)
            e = e.oPrev
        elif e.oNext == startingEdge:
            return
        else:
            e = e.oNext.lPrev

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
