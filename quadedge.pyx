# Quadedge implementation based on Lischinski's code published in
# Graphics Gems IV: https://github.com/erich666/GraphicsGems/tree/master/gemsiv/delaunay

from __future__ import division

import numpy as np
cimport numpy as np
from cpython cimport array as c_array
from array import array
import sys
from numbers import Number

# Use real division everywhere


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
    #cdef public long x, y
    #cdef public float z
    #cdef public Edge parent

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
    
    cpdef inTriangleSlow(self, Triangle tri):
        v0, v1, v2 = tri.vertices
        
        #s = 1.0 / (tri.area) * (v0.y * v2.x - v0.x * v2.y + (v2.y - v0.y) * self.x + (v0.x - v2.x) * self.y )
        #if s < 0:
        #    return False
        #else:
        #    t = 1.0 / (tri.area) * (v0.x * v1.y - v0.y * v1.x + (v0.y - v1.y) * self.x + (v1.x - v0.x) * self.y )
        #    if t < 0:
        #        return False
        #    else:
        #        return (s + t) < 1
        va = v2 - v0
        vb = v1 - v0
        vc = self - v0
        
        d00 = va * va
        d01 = va*vb
        d02 = va*vc
        d11 = vb*vb
        d12 = vb*vc
        
        denom = float(d00 * d11 - d01 * d01)
        u = (d11 * d02 - d01 * d12) / denom
        if u >= 0:
            v = (d00 * d12 - d01 * d02) / denom
            return u + v <= 1
        else:
            return False

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
        
        # print "Created triangle: ", self.__repr__(), self, "\n"
    
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
    
#    inline void Plane::init(const Vec3& p, const Vec3& q, const Vec3& r)
#// find the plane z=ax+by+c passing through three points p,q,r
#{
#    // We explicitly declare these (rather than putting them in a
#    // Vector) so that they can be allocated into registers.
#    real ux = q[X]-p[X], uy = q[Y]-p[Y], uz = q[Z]-p[Z];
#    real vx = r[X]-p[X], vy = r[Y]-p[Y], vz = r[Z]-p[Z];
#    real den = ux*vy-uy*vx;
#
#    a = (uz*vy - uy*vz)/den;
#    b = (ux*vz - uz*vx)/den;
#    c = p[Z] - a*p[X] - b*p[Y];
#}
    
    cpdef rasterize_triangle(self, np.ndarray[np.float_t, ndim=2] H):
        cdef float max_error = 0
        cdef int max_x, max_y
        cdef Vertex v0, v1, v2
        v0, v1, v2 = self.vertices
        
        if v0.y > v1.y:
            v_tmp = v0
            v0 = v1
            v1 = v_tmp
        if v0.y > v2.y:
            v_tmp = v0
            v0 = v2
            v2 = v_tmp
        if v1.y > v2.y:
            v_tmp = v1
            v1 = v2
            v2 = v_tmp
        if v1.y == v2.y:
            x, y, max_error = self.rasterize_flat(v0, v1, v2, H)
            #print "Triangle ", self, x, y, max_error
        elif v0.y == v1.y:
            x, y, max_error = self.rasterize_flat(v2, v0, v1, H)
            #print "Triangle ", self, x, y, max_error
        else:
            v_tmp = Vertex((v0.x + float(v1.y - v0.y) / float(v2.y - v0.y) \
                     * (v2.x - v0.x)), v1.y)
            #print "Aux point: ", v_tmp
            x, y, max_error = self.rasterize_flat(v0, v1, v_tmp, H)
            #print "Found new candidate (top): ", x, y, max_error
            x2, y2, max_error2 = self.rasterize_flat(v2, v1, v_tmp, H,
                                                     max_error = max_error,
                                                     include_last_row = False)
            
            #x = array('i', [-1] * (i1 + i2 + 1))
            #y = array('i', [-1] * (i1 + i2 + 1))
            #x[0  : i1     ] = x1
            #x[i1 : i1 + i2] = x2
            #x[i1 + i2]      = v_tmp.x
            
            #y[0  : i1     ] = y1
            #y[i1 : i1 + i2] = y2
            #y[i1 + i2]      = v_tmp.y
            
           # x = np.concatenate((x1, x2))
            #y = np.concatenate((y1, y2))
            if max_error2 > max_error:
                x, y, max_error = x2, y2, max_error2
                #print "Found new candidate (bottom): ", x, y, max_error
            
            x3 = round(v_tmp.x)
            y3 = round(v_tmp.y)
            error3 = abs(H[y3, x3] - self.interpolate(x3, y3))
            
            if error3 > max_error:
                x, y, max_error = x3, y3, error3
                #print "Found new candidate (aux): ", x, y, max_error 
        
        #print "Final: ", x, y, max_error, self
        return Vertex(x, y, H[y, x]), max_error
  
    def __str__(self):
        return "{} -- {} -- {}".format(self.vertices[0],
                                       self.vertices[1],
                                       self.vertices[2])
                                       
    cdef rasterize_flat(self, Vertex v0, Vertex v1, Vertex v2,
                        np.ndarray[np.float_t, ndim=2] H,
                        include_last_row = True,
                        float max_error = 0):
        if v1.x > v2.x:
            v1, v2 = v2, v1
        
        if v1.y > v0.y:
            step_y = 1
        else:
            step_y = -1
        
        cdef int max_elements = (v1.y - v0.y + 1) * (v2.x - v1.x + 1) * step_y
        
        cdef float slope1 = float(v1.x - v0.x) / float(v1.y - v0.y) * step_y
        cdef float slope2 = float(v2.x - v0.x) / float(v2.y - v0.y) * step_y
        cdef float x_start = v0.x + slope1
        cdef float x_stop = v0.x + slope2
        
        #cdef c_array.array x = array('i', [-1] * max_elements)
        #cdef c_array.array y = array('i', [-1] * max_elements)
        
        cdef np.ndarray[DTYPE_t, ndim=1] x = np.zeros(max_elements, dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1] y = np.zeros(max_elements, dtype=DTYPE)
        cdef int i = 0    
        
        cdef int max_x = 0, max_y = 0
        
        for scanline_y in xrange(v0.y + step_y, v1.y, step_y):
            for scanline_x in xrange(int(round(x_start)), int(round(x_stop)) + 1):
                x[i] = scanline_x
                y[i] = scanline_y
                error = abs(H[scanline_y, scanline_x] - self.interpolate(scanline_x, scanline_y))
                
                if error > max_error:
                    max_error = error
                    max_x = scanline_x
                    max_y = scanline_y
                    #print "Found new candidate: ", max_x, max_y, max_error 
                i += 1
            #n = int(x_stop) - int(x_start) + 1
            #x[i:i+n] = range(int(x_start), int(x_stop) + 1)
            #i += n
            x_start += slope1
            x_stop += slope2
        
        if include_last_row:
            scanline_y = v1.y
           # print x_start, x_stop
            for scanline_x in xrange(int(round(x_start)), int(round(x_stop))):
                error = abs(H[scanline_y, scanline_x] - self.interpolate(scanline_x, scanline_y))
                
                if error > max_error:
                    max_error = error
                    max_x = scanline_x
                    max_y = scanline_y
                    #print "Found new candidate: ", max_x, max_y, max_error 
                i += 1
        
        return max_x, max_y, max_error




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
    cdef Edge base, t
    
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