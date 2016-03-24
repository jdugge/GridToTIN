# Quadedge implementation based on Lischinski's code published in
# Graphics Gems IV: https://github.com/erich666/GraphicsGems/tree/master/gemsiv/delaunay

import numpy as np
import sys
from numbers import Number

import math


eps = 1e-6
float_min = -sys.float_info.max


class Edge:
    def __init__(self, origin=None):
        self.origin_ = origin
        if origin is not None:
            # Add a reference from the origin vertex to this edge, so if we need
            # an edge that contains a certain vertex, we can simply use this
            # reference
            origin.edge = self
        self.rot = self
        self.next = self

    def __str__(self):
        return "(" + str(self.origin.x) + "," + str(self.origin.y) + \
            ") -- (" + str(self.destination.x) + "," + str(self.destination.y) + ")"

    @property
    def origin(self): return self.origin_

    @origin.setter
    def origin(self, origin):
        self.origin_ = origin
        origin.edge = self

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
        return not self.o_prev.destination.right_of(self)

    def as_line_segment(self):
        return [self.origin.pos, self.destination.pos]

    def selection_segment(self, v):
        """
        When the edge is encroached by a vertex v, there is a segment of the
        edge along which the edge can be split to solve the encroachment, the
        "selection segment".
        :param v: Encroaching vertex
        :return: The start and end vertices of the selection segment
        """
        a = self.origin
        b = self.destination

        # Find the start of the selection segment
        av = v - a
        ab = b - a

        p = min(av.norm**2/(av * ab), 1)
        p = p if p >= 0 else 1
        s0 = a + p * ab

        # Find the end of the selection segment
        bv = v - b
        ba = a - b

        q = bv.norm**2/(bv * ba)
        q = q if q >= 0 else 1
        s1 = b + q * ba

        return s0, s1

    def __lt__(self, other):
        self.length < other.length
    

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
        self.edge = None

    @property
    def pos(self): return self.x, self.y, self.z

    @pos.setter
    def pos(self, pos):
        self.x = pos[0]
        self.y = pos[1]
        self.z = pos[2]

    def __str__(self):
        return "({},{},{})".format(*self.pos)

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

    def __rmul__(self, other):
        if isinstance(other, Number):
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

    def encroaches(self, e):
        """
        Checks if the vertex encroaches a given edge, that is it lies within
        the diametral circle of the edge and is
        :param e:
        :return: True if the vertex encroaches e, False otherwise
        """
        if self is e.origin or self is e.destination:
            return False
        else:
            a = e.origin - self
            b = e.destination - self
            return a * b <= 0

    @property
    def star(self):
        start = e = self.edge
        edges = [start]
        while e.o_next is not start:
            e = e.o_next
            edges.append(e)
        return edges

class Triangle:
    def __init__(self, e, anchor=True, id_=-1):
        self.vertices = [e.origin, e.destination, e.l_prev.origin]
        self.area = triangle_area(*self.vertices)
        self.id = id_
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
        self.c = self.vertices[0].z - self.a * self.vertices[0].x - self.b * self.vertices[0].y

    def interpolate(self, x, y):
        return self.a * x + self.b * y + self.c
    
    def find_circumcenter(self,  triangulation, within_triangulation=False):
            v0 = self.anchor.destination - self.anchor.origin
            v1 = self.anchor.l_next.destination - self.anchor.origin
            
            D = 2 * (v0.x * v1.y - v0.y * v1.x)
            c_x = ( v1.y * (v0.x**2 + v0.y**2) - v0.y * (v1.x**2 + v1.y**2) ) / D
            c_y = ( v0.x * (v1.x**2 + v1.y**2) - v1.x * (v0.x**2 + v0.y**2) ) / D
            
            circumcenter = Vertex(c_x, c_y) + self.anchor.origin
            
            if within_triangulation:
                boundary_edge = None
                if circumcenter.x > triangulation.maxX:
                    circumcenter.x = triangulation.maxX
                    boundary_edge = triangulation.locate(circumcenter)
                elif circumcenter.x < triangulation.minX:
                    circumcenter.x = triangulation.minX
                    boundary_edge = triangulation.locate(circumcenter)
                elif circumcenter.y > triangulation.maxY:
                    circumcenter.y = triangulation.maxY
                    boundary_edge = triangulation.locate(circumcenter)
                elif circumcenter.y < triangulation.minY:
                    circumcenter.y = triangulation.minY
                    boundary_edge = triangulation.locate(circumcenter)
                
                if boundary_edge is not None:
                    circumcenter = (boundary_edge.origin +
                                    boundary_edge.destination) / 2
                
            return circumcenter

    def offcenter(self, b=math.sqrt(2)):
        """
        Find the off-center of the triangle, using the definition given in
        Üngör, Alper. "Off-centers: A new type of Steiner points for computing
        size-optimal quality-guaranteed Delaunay triangulations."
        Computational Geometry 42.2 (2009): 109-118.
        :param b: Minimum radius-edge ratio. Defaults to sqrt(2), as suggested
                  in Üngör's paper
        :return: A vertex at the 2D position of the off-center
        """
        # Get the shortest edge
        e = min(self.edges)

        # Assign some shorter names
        vo = e.origin
        vd = e.destination
        va = e.o_next.destination

        # Subtract e.origin
        do = vd - vo
        ao = va - vo

        # Project b onto the perpendicular bisector of e using Pythagoras
        b_proj = math.sqrt(b**2 - 0.25)

        # The off-center
        dx_offcenter = 0.5 * do.x - b_proj * do.y
        dy_offcenter = 0.5 * do.y + b_proj * do.x

        # The circumcenter
        denominator = 0.5 / (do.x * ao.y - ao.x * do.y)
        dx_circumcenter = (ao.y * do.norm**2 - do.y * ao.norm**2) * denominator
        dy_circumcenter = (do.x * ao.norm**2 - ao.x * do.norm**2) * denominator

        # Choose the center that is closer to e.origin
        if dx_offcenter**2 + dy_offcenter**2 < \
           dx_circumcenter**2 + dy_circumcenter**2:
            dx, dy = dx_offcenter, dy_offcenter
        else:
            dx, dy = dx_circumcenter, dy_circumcenter

        return Vertex(vo.x + dx, vo.y + dy)

    @property
    def coordinate_list(self):
        return [v.pos for v in self.vertices]

    @property
    def coordinate_list_2d(self):
        return [v.pos[:2] for v in self.vertices]

    @property
    def edge_lengths(self):
        l0 = (self.vertices[0] - self.vertices[1]).norm
        l1 = (self.vertices[1] - self.vertices[2]).norm
        l2 = (self.vertices[2] - self.vertices[0]).norm
        l = [l0, l1, l2]
        l.sort()

        return l

    @property
    def aspect_ratio(self):
        l0, l1, l2 = self.edge_lengths
        return l2 * (l0 + l1 + l2) / self.area

    @property
    def circumradius(self):
        l0, l1, l2 = self.edge_lengths
        return l0 * l1 * l2 / (2 * self.area)

    @property
    def radius_edge_ratio(self):
        l0, l1, l2 = self.edge_lengths
        return self.circumradius / l0

    @property
    def edges(self):
        return [self.anchor, self.anchor.l_next, self.anchor.l_prev]

    def __str__(self):
        return "{} -- {} -- {}".format(self.vertices[0],
                                       self.vertices[1],
                                       self.vertices[2])


def triangle_area(v0, v1, v2):
    return (v1.x - v0.x) * (v2.y - v0.y) - (v1.y - v0.y) * (v2.x - v0.x)


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
