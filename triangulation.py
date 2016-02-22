# Implementation of Garland and Heckbert's sequential greedy insertion
# algorithm for terrain approximation: http://mgarland.org/software/terra.html


from quadedge import Edge, Vertex, splice, connect, swap, make_triangle, delete_edge, make_edge, Triangle
import copy
import numpy as np
from heap import Heap
from math import floor, ceil
import rasterio
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from matplotlib.collections import PatchCollection


class Triangulation:
    def __init__(self, DEM):
        if isinstance(DEM, np.ndarray):
            H = DEM
            self.Hmap = H
        elif isinstance(DEM, str):
            with rasterio.drivers():
                with rasterio.open(DEM) as src:
                    rawdata = src.read()
                    H = np.array(rawdata.squeeze()[50:-50, 50:-50], dtype=float)
                    self.Hmap = H


        minX = 0
        minY = 0
        maxX = self.Hmap.shape[1] - 1
        maxY = self.Hmap.shape[0] - 1

        self.heap = Heap()
        self.triangle_list = []

#        assert maxX > minX
#        assert maxY > minY
#
        self.minX = minX
        self.minY = minY
        self.maxX = maxX
        self.maxY = maxY

        rangeX = maxX - minX
        rangeY = maxY - minY

        self.vertex_dict = dict()
        self.edge_dict = dict()

        v0 = Vertex(minX, minY, self.Hmap[0,   0])
        v1 = Vertex(maxX, minY, self.Hmap[0,  -1])
        v2 = Vertex(maxX, maxY, self.Hmap[-1, -1])
        v3 = Vertex(minX, maxY, self.Hmap[-1,  0])

        self.vertex_dict[0] = v0
        self.vertex_dict[1] = v1
        self.vertex_dict[2] = v2
        self.vertex_dict[3] = v3
        self.nextVertexid = 4

        self.next_edge_id = 0
        q0 = make_edge(v0, v1)
        q1 = make_edge(v2, v3)
        q2 = make_edge(v3, v0)
        q3 = make_edge(v1, v2)
        q4 = make_edge(v1, v3)

        splice(q0.sym, q4)
        splice(q4.sym, q2)
        splice(q2.sym, q0)
        splice(q0.sym, q3)
        splice(q3.sym, q1)
        splice(q1.sym, q4.sym)

        self.add_edge(q0)
        self.add_edge(q1)
        self.add_edge(q2)
        self.add_edge(q3)
        self.add_edge(q4)

        self.base = q0

        self.history = Triangle(self.base, anchor = False, id_= -1)
        self.history.children = [Triangle(q4),
                                 Triangle(q4.sym)]

        for triangle in self.history.children:
            self.triangle_list.append(triangle)
            self.scan_triangle(triangle)
            print(triangle.candidate_error, triangle.candidate)
            triangle.id = self.heap.insert(triangle.candidate_error,
                                           (triangle.candidate, triangle))

    @property
    def vertices(self): return [self.vertex_dict[key]
                                for key in self.vertex_dict]

    @property
    def edges(self): return [self.edge_dict[key]
                             for key in self.edge_dict]

    @property
    def triangles(self):
            return list(filter(lambda x: x.id != -1, self.triangle_list))

    def triangle_patches(self, **kwargs):
            return PatchCollection([
                Polygon([v.pos[:2] for v in t.vertices])
                for t in self.triangles
            ], **kwargs)

    @property
    def edge_lines(self):
            return

    # The walking method for finding a triangle. Not used in the main code, but
    # kept because it's pretty to look at
    def locate(self, v):
        e = self.base
        while True:
            if v == e.origin or v == e.destination or v.on_edge(e):
                # print "Start or end of edge"
                if not e.o_next.destination.left_of(e):
                    # The mesh is on the right-hand side of the edge, flip it
                    e = e.sym
                return e
            elif v.right_of(e):
                # print "Flip"
                e = e.sym
            elif not (v.right_of(e.o_next)):
                # print "Left of o_next, move to o_next"
                e = e.o_next
            elif not (v.right_of(e.d_prev)):
                # print "Left of dPrev, move to dPrev"
                e = e.d_prev
            else:
                # print "Found the triangle"
                return e

    # Point location using the history graph. Much faster than the walking
    # method
    def search(self, v):
        current_triangle = self.history

        while len(current_triangle.children) > 0:
            for triangle in current_triangle.children:
                if v.in_triangle(triangle.vertices[0],
                                 triangle.vertices[1],
                                 triangle.vertices[2]):
                    current_triangle = triangle
                    break
            if not current_triangle == triangle:
                # None of the children contained the point, point is not in
                # triangulation
                return None
        return current_triangle.anchor

    def add_edge(self, e):
        self.edge_dict[self.next_edge_id] = e
        e.id = e.sym.id = self.next_edge_id
        self.next_edge_id += 1

    def delete_edge(self, e):
        splice(e, e.o_prev)
        splice(e.sym, e.sym.o_prev)
        del self.edge_dict[e.id]
        del e

    def insert_site(self, v, e=None):
        deleted_triangles = []
        created_triangles = []
        boundary_edge = None

        if e is None:
            e = self.search(v)
        else:
            assert (not v.right_of(e) and
                    not v.right_of(e.lNext) and
                    not v.right_of(e.l_prev) ), \
                    'Edge %s is not an edge of the ' \
                    'triangle containing edge %s' % (e, v)
        if v == e.origin or v == e.destination:
            return deleted_triangles, created_triangles
        elif v.on_edge(e):
            if not e.o_prev.destination.right_of(e):
                parents = [e.triangle]
                boundary_edge = e
            else:
                parents = [e.triangle, e.sym.triangle]
                e = e.o_prev
                self.delete_edge(e.o_next)
        else:
            parents = [e.triangle]

        # Add point to triangulation
        self.vertex_dict[self.nextVertexid] = v
        self.nextVertexid += 1

        # Create first spoke from origin of base to new site
        spoke = make_edge(e.origin, v)
        self.add_edge(spoke)

        splice(spoke, e)
        starting_spoke = spoke

        # Create second spoke from destination of base to new site
        spoke = connect(e, spoke.sym)
        self.add_edge(spoke)

        e = spoke.o_prev
        while e.l_next != starting_spoke:
            spoke = connect(e, spoke.sym)
            self.add_edge(spoke)
            e = spoke.o_prev

        if boundary_edge is not None:
            # We might be deleting the base edge of the mesh, so use an edge
            # that is guaranteed to survive
            self.base = e
            self.delete_edge(boundary_edge)

        # Create (potentially ephemeral) triangles for all the spokes and assign
        # them as children to the parent triangles
        current_spoke = starting_spoke
        while True:
            current_spoke = current_spoke.d_next
            if current_spoke.o_next.destination.left_of(current_spoke):
                child = Triangle(current_spoke)
                created_triangles.append(child)
                for parent in parents:
                    parent.children.append(child)

            if current_spoke == starting_spoke:
                break

        for parent in parents:
            parent.anchor = None
        deleted_triangles.extend(parents)

        while True:
            t = e.o_prev
            if t.destination.right_of(e) and v.in_circle(e.origin,
                                                         t.destination,
                                                         e.destination):
                parents = [e.triangle, e.sym.triangle]
                swap(e)
                deleted_triangles.extend(parents)

                children = [Triangle(e),
                            Triangle(e.sym)]
                created_triangles.extend(children)
                for parent in parents:
                    parent.children.extend(children)
                    parent.anchor = None

                e = e.o_prev
            elif e.o_next == starting_spoke:
                break
            else:
                e = e.o_next.l_prev

        created_triangles = list(set(created_triangles) -
                                 set(deleted_triangles))

        return created_triangles, deleted_triangles

    def find_nearest_boundary_edge(self, vertex):
        # Let's find a boundary edge
        start_edge = self.search(Vertex(0, 0))
        edge = start_edge.o_next
        while not edge == start_edge:
            if edge.is_boundary:
                break
            edge = edge.o_next
        if not edge.is_boundary:
            start_edge = edge.sym
            edge = start_edge.o_next
            while not edge == start_edge:
                if edge.is_boundary:
                    break
                edge = edge.o_next

        while (not 0 < ((vertex - edge.origin) *
               (edge.destination - edge.origin)) / edge.length**2 < 1) \
                or not vertex.left_of(edge):
            edge = edge.lNext

    def scan_triangle(self, t):
        v0, v1, v2 = t.vertices

        # Sort vertices in ascending order
        if v0.y > v1.y:
            v0, v1 = v1, v0
        if v0.y > v2.y:
            v0, v2 = v2, v0
        if v1.y > v2.y:
            v1, v2 = v2, v1


        # Check if base of triangle is flat
        if v1.y == v0.y:
            dx0 = 0.0
        else:
            dx0 = (v1.x - v0.x) / (v1.y - v0.y)

        dx1 = (v2.x - v0.x) / (v2.y - v0.y)

        y_start = v0.y
        y_end   = v1.y
        x_a     = v0.x
        x_b     = v0.x

        # If the base of the triangle is flat, this loop won't be executed
        for y in range(y_start, y_end):
            self.scan_triangle_line(t, y, x_a, x_b)
            x_a += dx0
            x_b += dx1

        # We have reached the second half of the triangle
        # Check if top of triangle is flat
        if v2.y == v1.y:
            dx0 = 0.0
        else:
            dx0 = (v2.x - v1.x) / (v2.y - v1.y)

        y_start = v1.y
        y_end   = v2.y
        x_a     = v1.x

        # If the top of the triangle is flat, this loop will be executed once
        for y in range(y_start, y_end + 1):
            self.scan_triangle_line(t, y, x_a, x_b)
            x_a += dx0
            x_b += dx1

    def scan_triangle_line(self, t, y, x_a, x_b):
        x_start = int(ceil(min(x_a, x_b)))
        x_end   = int(floor(max(x_a, x_b)))

        for x in range(x_start, x_end + 1):
            z_map = self.Hmap[y, x]
            error = abs(z_map - t.interpolate(x, y))
            if error > t.candidate_error:
                t.candidate_error = error
                t.candidate.pos = (x, y, z_map)

    def insert_point(self, v):
        new, deleted = self.insert_site(v)

        for triangle in deleted:
            triangle.id = -1

        for triangle in new:
            triangle.id = -2

        #for triangle in new:
         #       self.scan_triangle(triangle)
                #triangle.id = self.heap.insert(triangle.candidate_error,
                #                               (triangle.candidate, triangle))
        self.triangle_list.extend(new)

    def insert_next(self):
        error, (candidate, triangle) = self.heap.pop()
        triangle.id = -1 # Mark it as removed from the heap

        new, deleted = self.insert_site(candidate)

        for triangle in deleted:
            if not triangle.id == -1:
                self.heap.delete(triangle.id)
                triangle.id = -1

        for triangle in new:
                self.scan_triangle(triangle)
                triangle.id = self.heap.insert(triangle.candidate_error,
                                               (triangle.candidate, triangle))
        self.triangle_list.extend(new)
