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
import logging
logging.basicConfig(level=logging.DEBUG)


class Triangulation:
    def __init__(self, dem, minimum_gap=5):
        if isinstance(dem, np.ndarray):
            self.dem = dem
        elif isinstance(dem, str):
            with rasterio.drivers():
                with rasterio.open(dem) as src:
                    rawdata = src.read()
                    self.dem = np.array(rawdata.squeeze(), dtype=float)

        self.minimum_gap = minimum_gap

        min_x = 0
        min_y = 0
        max_x = self.dem.shape[1] - 1
        max_y = self.dem.shape[0] - 1

        self.heap = Heap()
        self.triangle_list = []

        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y

        self.available = np.ones_like(self.dem)

        self.vertex_dict = dict()
        self.edge_dict = dict()

        v0 = Vertex(min_x, min_y, self.dem[0, 0])
        v1 = Vertex(max_x, min_y, self.dem[0, -1])
        v2 = Vertex(max_x, max_y, self.dem[-1, -1])
        v3 = Vertex(min_x, max_y, self.dem[-1, 0])

        self.vertex_dict[0] = v0
        self.vertex_dict[1] = v1
        self.vertex_dict[2] = v2
        self.vertex_dict[3] = v3
        self.next_vertex_id = 4

        self.next_edge_id = 0
        # Boundary rectangle
        q0 = make_edge(v0, v1)
        q1 = make_edge(v2, v3)
        q2 = make_edge(v3, v0)
        q3 = make_edge(v1, v2)
        # Diagonal
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

        # Mark area around border edges as unavailable
        for e in self.edges:
            self.mark_availability(e.origin,
                                   e.destination,
                                   radius=self.minimum_gap,
                                   value=0)
        # Mark area on border edges themselves as available
        for e in self.edges:
            self.mark_availability(e.origin,
                                   e.destination,
                                   radius=0,
                                   value=1)
        # Mark area around border vertices as unavailable
        for v in self.vertices:
            self.mark_availability(v, radius=minimum_gap, value=0)

        self.add_edge(q4)

        self.base = q0

        self.history = Triangle(self.base, anchor = False, id_= -1)
        self.history.children = [Triangle(q4),
                                 Triangle(q4.sym)]

        for triangle in self.history.children:
            self.triangle_list.append(triangle)
            self.scan_triangle(triangle)
            triangle.id = self.heap.insert(triangle.candidate_error,
                                           (triangle.candidate, triangle))

    @property
    def vertices(self): return [self.vertex_dict[key]
                                for key in self.vertex_dict]

    @property
    def edges(self): return [self.edge_dict[key]
                             for key in self.edge_dict]

    @property
    def undirected_edges(self):
        edges = set()
        for triangle in self.triangles:
            for edge in triangle.edges:
                if edge not in edges and edge.sym not in edges:
                    edges.add(edge)
        return edges

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
                # print "Left of d_prev, move to d_prev"
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
        """
        Insert a new site in the 2D triangulation, maintaining the Delaunay
        criterion
        :param v: Vertex to be inserted into the triangulation
        :param e: Optional: edge of the triangle that contains v, to speed up
        the process
        :return:
        """
        deleted_triangles = []
        created_triangles = []
        boundary_edge = None

        # Get elevation from map if none was provided in vertex
        if v.z == 0:
            v.z = self.dem[v.y, v.x]

        if e is None:
            e = self.search(v)
        else:
            assert (not v.right_of(e) and
                    not v.right_of(e.l_next) and
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
        self.vertex_dict[self.next_vertex_id] = v
        self.next_vertex_id += 1

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
                # Delaunay criterion is violated, swap an edge to fix it
                # This deletes two triangles and creates two new ones
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

        self.mark_availability(v, radius=self.minimum_gap, value=0)
        return created_triangles, deleted_triangles

    def find_nearest_boundary_edge(self, vertex):
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
            edge = edge.l_next

    def scan_triangle(self, t, interpolation_map=None, only_return_points=False):
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

        points = []
        # If the base of the triangle is flat, this loop won't be executed
        for y in range(y_start, y_end):
            points.extend(self.scan_triangle_line(t, y,
                                                  x_a, x_b,
                                                  interpolation_map,
                                                  only_return_points))
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
            points.extend(self.scan_triangle_line(t, y,
                                                  x_a, x_b,
                                                  interpolation_map,
                                                  only_return_points))
            x_a += dx0
            x_b += dx1

        if only_return_points:
            return points

    def scan_triangle_line(self, t, y, x_a, x_b, interpolation_map=None,
                           only_return_points = False):
        x_start = int(ceil(min(x_a, x_b)))
        x_end   = int(floor(max(x_a, x_b)))

        points = []
        for x in range(x_start, x_end + 1):
            if only_return_points:
                points.append((x,y))
            else:
                z_map = self.dem[y, x]
                error = abs(z_map - t.interpolate(x, y))
                if interpolation_map is not None:
                    interpolation_map[y,x] = t.interpolate(x,y)
                if error > t.candidate_error and self.available[y, x] == 1:
                    t.candidate_error = error
                    t.candidate.pos = (x, y, z_map)
        return points

    def scan_segment(self, e, s0, s1):
        """
        Find the point along the selection segment of an edge that has the worst
        discretisation error
        :param e: edge containing the segment
        :param s0: starting vertex of selection segment
        :param s1: ending vertex of selection segment
        :return: vertex at the 2D position with the worst discretisation error
        """

        a = s1 - s0
        d = ceil(a.norm)

        step = 1 / d

        max_error = 0
        worst_vertex = None

        for i in range(d):
            v = s0 + i * step * a
            x = int(round(v.x))
            y = int(round(v.y))

            if self.available[y, x] == 1:
                z_map = self.dem[y, x]
                interpolation = e.triangle.interpolate(x, y)
                error = abs(interpolation - z_map)
                if error > max_error:
                    max_error = error
                    worst_vertex = Vertex(x, y)
        return worst_vertex

    def circle_points(self, center, radius):
        circle_points = []
        y_start = (max(round(center.y - radius), self.min_y))
        y_end = (min(round(radius + 1 + center.y), self.max_y + 1))

        for y in range(y_start, y_end):
            x_max = max(radius**2 - (y-center.y)**2, 0)**0.5
            x_start = int(max(round(center.x - x_max), self.min_x))
            x_end = int(min(round(x_max + 1 + center.x), self.max_x))
            for x in range(x_start, x_end + 1):
                circle_points.append((x,y))

        return circle_points

    def segment_points(self, s0, s1):
        """
        Find all points along the selection segment of an edge
        :param s0: starting vertex of segment
        :param s1: ending vertex of segment
        :return: list of 2D coordinates along segment
        """
        segment_points = []
        a = s1 - s0
        d = ceil(a.norm)

        step = 1 / d

        for i in range(d):
            v = s0 + i * step * a
            x = int(round(v.x))
            y = int(round(v.y))
            segment_points.append((x, y))
        return segment_points

    def mark_availability(self, v0, v1=None, radius=0, value=0):
        if v1 is not None:
            segment_points = self.segment_points(v0, v1)
        else:
            segment_points = [(v0.x, v0.y)]

        for s in segment_points:
            cp = self.circle_points(Vertex(s[0], s[1]), radius)
            for p in cp:
                self.available[p[1], p[0]] = value

    def scan_circle(self, center, radius):
        max_error = 0
        worst_vertex = None

        y_start = (max(round(center.y - radius), self.min_y))
        y_end = (min(round(radius + 1 + center.y), self.max_y))

        for y in range(y_start, y_end):
            x_max = max(radius**2 - round(y-center.y)**2, 0)**0.5
            x_start = int(max(round(center.x - x_max), self.min_x))
            x_end = int(min(round(x_max + 1 + center.x), self.max_x))
            for x in range(x_start, x_end):
                if self.available[y, x] == 1:
                    triangle = self.search(center).triangle

                    z_map = self.dem[y, x]
                    interpolation = triangle.interpolate(x, y)
                    error = abs(interpolation - z_map)
                    if error > max_error:
                        max_error = error
                        worst_vertex = Vertex(x, y)

        return worst_vertex

    def insert_point(self, v, e=None):
        """
        Insert a new vertex into the triangulation and scan the newly created
        triangles for the error
        :param v: Vertex to be inserted
        :param e: Optional edge for starting the triangle search
        :return:
        """
        new, deleted = self.insert_site(v, e)

        for triangle in deleted:
            if not triangle.id == -1:
                self.heap.delete(triangle.id)
                triangle.id = -1

        for triangle in new:
            triangle.id = -2

        for triangle in new:
            self.scan_triangle(triangle)
            triangle.id = self.heap.insert(triangle.candidate_error,
                                           (triangle.candidate, triangle))
        self.mark_availability(v, radius=self.minimum_gap, value=0)
        self.triangle_list.extend(new)

    def insert_next(self):
        """
        Pop the candidate with the greatest error and insert it into the
        triangulation
        :return:
        """
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

    def split_edge(self, e):
        """
        Insert a new vertex at the mid-point of the given edge
        :param e: The edge to split
        :return:
        """
        new_v = e.origin + (e.destination - e.origin) * 0.5
        new_v.x = int(new_v.x)
        new_v.y = int(new_v.y)
        new_v.z = 0
        if new_v.right_of(e):
            e = e.sym
        self.insert_point(new_v, e)

    def worst_encroached_edge(self):
        """
        :return: The longest encroached edge and the encroaching vertex
        """
        while True:
            max_length = 0
            worst_edge = None
            encroaching_vertex = None
            for e in self.undirected_edges:
                encroaching_vertex_candidate = None
                if e.is_boundary or e.sym.is_boundary:
                    p = e.o_next.destination
                    q = e.o_prev.destination
                    if p.encroaches(e):
                        encroaching_vertex_candidate = p
                    elif q.encroaches(e):
                        encroaching_vertex_candidate = q

                    if encroaching_vertex_candidate is not None:
                        if e.length > max_length:
                            max_length = e.length
                            worst_edge = e
                            encroaching_vertex = encroaching_vertex_candidate

            if worst_edge is None:
                logging.debug("No more encroached edges")
                return None, None
            else:
                return worst_edge, encroaching_vertex

    def split_all_encroached_edges(self):
        """
        For all boundary edges, check if they are encroached. Split the longest
        encroached boundary edge and repeat until no more boundary edges are
        encroached
        """
        while True:
            worst_edge, v = self.worst_encroached_edge()
            if worst_edge is None:
                break
            else:
                logging.debug("Splitting edge {}".format(worst_edge))
                s0, s1 = worst_edge.selection_segment(v)
                split = self.scan_segment(worst_edge, s0, s1)
                self.insert_point(split)

    def interpolated_map(self):
        """
        The height map resulting from linear interpolation of the triangle
        mesh
        :return:
        """
        interpolated_map = self.dem.copy()
        for triangle in self.triangles:
            self.scan_triangle(triangle, interpolated_map)
        return interpolated_map

    def error_map(self):
        """
        The difference map between the original height map and the interpolated
        height map
        :return:
        """
        error_map = self.dem.copy() - self.interpolated_map()
        return error_map

    def bad_triangles(self, b=2**0.5, ):
        """
        Find all triangles with a circumradius-to-shortest-edge ratio smaller
        than b
        :param b: Threshold, sqrt(2) by default
        :return: A list of all bad triangles, or None if no bad triangles exist
        """
        bad_triangles = [triangle for triangle in self.triangles
                          if (triangle.radius_edge_ratio > b)]
        if len(bad_triangles) == 0:
            return None
        else:
            bad_triangles.sort(key=lambda x: x.radius_edge_ratio)
            return bad_triangles

    def fix_worst_triangle(self, worst_triangle=None, use_selection_disk=False):
        """
        Find the worst triangle (in terms of circumradius-to-shortest-edge
        ratio) and try to insert its off-center. If the off-center would
        encroach a boundary edge, split that edge instead of inserting the
        off-center
        :param worst_triangle: Optionally provide the triangle that should be
                               split instead of finding the worst triangle
        :return:
        """
        if worst_triangle is None:
            bad_triangles = self.bad_triangles()
            if len(bad_triangles) == 0:
                logging.debug("No bad triangle found")
                return None
            else:
                worst_triangle = bad_triangles[0]

        logging.debug("Fixing bad triangle {}".format(worst_triangle))
        if use_selection_disk:
            c, r = worst_triangle.selection_disk()
            v = self.scan_circle(c, r)
        else:
            v = worst_triangle.offcenter()

        v.x = int(v.x)
        v.y = int(v.y)

        logging.debug("Trying to insert off-center {}".format(v))
        encroachment = False
        for e in self.boundary_edges():
            if v.encroaches(e):
                encroachment = True
                logging.debug("Off-center would encroach {}".format(e) +
                              ", splitting edge instead")
                self.split_edge(e)

        if not encroachment:
            logging.debug("Inserting off-center")
            self.insert_point(v)

    def fix_all_bad_triangles(self, use_selection_disk=False):
        """
        While there are still bad triangles, insert Steiner points and split
        encroached edges
        """
        bad_triangles = self.bad_triangles()
        while bad_triangles is not None:
            worst_triangle = bad_triangles[0]
            self.fix_worst_triangle(worst_triangle, use_selection_disk)
            self.split_all_encroached_edges()
            bad_triangles = self.bad_triangles()
        logging.debug("No bad triangles")

    def boundary_edges(self):
        """
        :return: List of all edges on the domain boundary
        """
        edges = list(self.undirected_edges)
        boundary_edges = []

        for e in edges:
            if e.is_boundary:
                boundary_edges.append(e)
            elif e.sym.is_boundary:
                boundary_edges.append(e.sym)

        return boundary_edges
