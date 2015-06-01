# Implementation of Garland and Heckbert's sequential greedy insertion
# algorithm for terrain approximation: http://mgarland.org/software/terra.html

from __future__ import division

from quadedge cimport Edge, Vertex, splice, connect, swap, makeTriangle, deleteEdge, makeEdge, Triangle
import copy
import numpy as np
cimport numpy as np
from heap import Heap
from libc.math cimport floor, ceil

DTYPE = np.float
ctypedef np.float_t DTYPE_t

cimport cython
@cython.boundscheck(False)

cdef class Triangulation:
    cdef int nextTriangleID
    cdef int nextEdgeID
    cdef int nextVertexID
    cdef public long minX, maxX, minY, maxY
    cdef public Edge base
    cdef public dict vertexDict
    cdef public dict edgeDict
    cdef public Vertex va, vb, vc
    cdef public Triangle history
    cdef public np.ndarray Hmap
    cdef heap
    
    def __cinit__(self, np.ndarray[DTYPE_t, ndim=2] H):
        self.Hmap = H
        cdef int minX = 0
        cdef int minY = 0
        cdef int maxX = self.Hmap.shape[1] - 1
        cdef int maxY = self.Hmap.shape[0] - 1
        
        self.heap = Heap()
        
#        assert maxX > minX
#        assert maxY > minY
#        
#        self.minX = minX
#        self.minY = minY
#        self.maxX = maxX
#        self.maxY = maxY
        
        cdef int rangeX = maxX - minX
        cdef int rangeY = maxY - minY
        
        #self.va = Vertex(minX - 1, minY - 1)
        #self.vb = Vertex(minX + 3 * rangeX + 1, minY - 1)
        #self.vc = Vertex(minX - 1, minY + 2 * rangeY + 1)
        #self.base = makeTriangle(self.va, self.vb, self.vc)
        #self.auxVertices = [self.va, self.vb, self.vc]
        
        self.vertexDict = dict()
        self.edgeDict = dict()
        
        cdef Vertex v0 = Vertex(minX, minY, self.Hmap[0,   0])
        cdef Vertex v1 = Vertex(maxX, minY, self.Hmap[0,  -1])
        cdef Vertex v2 = Vertex(maxX, maxY, self.Hmap[-1, -1])
        cdef Vertex v3 = Vertex(minX, maxY, self.Hmap[-1,  0])
        
        self.vertexDict[0] = v0
        self.vertexDict[1] = v1
        self.vertexDict[2] = v2
        self.vertexDict[3] = v3
        self.nextVertexID = 4
        
        cdef Edge q0 = makeEdge(v0, v1)
        cdef Edge q1 = makeEdge(v2, v3)
        cdef Edge q2 = makeEdge(v3, v0)
        cdef Edge q3 = makeEdge(v1, v2)
        cdef Edge q4 = makeEdge(v1, v3)
        
        splice(q0.sym, q4)
        splice(q4.sym, q2)
        splice(q2.sym, q0)
        splice(q0.sym, q3)
        splice(q3.sym, q1)
        splice(q1.sym, q4.sym)
        
        self.addEdge(q0)
        self.addEdge(q1)
        self.addEdge(q2)
        self.addEdge(q3)
        self.addEdge(q4)
        
        self.base = q0
        
        self.history = Triangle(self.base, anchor = False, ID = -1)
        self.history.children = [Triangle(q4),
                                 Triangle(q4.sym)]
        
        for triangle in self.history.children:
            self.scan_triangle(triangle)
            triangle.ID = self.heap.insert(triangle.candidate_error,
                                           (triangle.candidate, triangle) )
        #self.nextTriangleID = 2
    
    property vertices:
        def __get__(self): return [self.vertexDict[key]
            for key in self.vertexDict]

    property edges:
        def __get__(self): return [self.edgeDict[key]
            for key in self.edgeDict]
    
    # The walking method for finding a triangle. Not used in the main code, but
    # kept because it's pretty to look at
    def locate(self, Vertex v):
        cdef Edge e = self.base
        while (True):
            if (v == e.origin or v == e.destination or v.onEdge(e)):
                # print "Start or end of edge"
                if not e.oNext.destination.leftOf(e):
                    # The mesh is on the right-hand side of the edge, flip it
                    e = e.sym
                return e
            elif v.rightOf(e):
                # print "Flip"
                e = e.sym
            elif not (v.rightOf(e.oNext)):
                # print "Left of oNext, move to oNext"
                e = e.oNext
            elif not (v.rightOf(e.dPrev)):
                # print "Left of dPrev, move to dPrev"
                e = e.dPrev
            else:
                # print "Found the triangle"
                return e
    
    # Point location using the history graph. Much faster than the walking
    # method
    def search(self, Vertex v):
        cdef Triangle current_triangle = self.history

        while len(current_triangle.children) > 0:
            for triangle in current_triangle.children:
                if v.inTriangle(triangle.vertices[0],
                                    triangle.vertices[1],
                                    triangle.vertices[2]):
                    current_triangle = triangle
                    break
        return current_triangle.anchor
    
    def addEdge(self, Edge e):
        self.edgeDict[self.nextEdgeID] = e
        e.ID = e.sym.ID = self.nextEdgeID
        self.nextEdgeID += 1
    
    def deleteEdge(self, Edge e):
        splice(e, e.oPrev)
        splice(e.sym, e.sym.oPrev)
        del self.edgeDict[e.ID]
        del e
    
    def insertSite(self, Vertex v, Edge e = None):
        cdef Edge boundary_edge = None
        cdef list deleted_triangles = []
        cdef list created_triangles = []
        cdef list parents
        cdef list children
        cdef Edge t
        
        if e is None:
            e = self.search(v)
        else:
            assert (not v.rightOf(e) and
                    not v.rightOf(e.lNext) and
                    not v.rightOf(e.lPrev) ), \
                    'Edge %s is not an edge of the ' \
                    'triangle containing edge %s' % (e, v)
        if v == e.origin or v == e.destination:
            return deleted_triangles, created_triangles
        elif v.onEdge(e):
            #print "On edge"
            if not e.oPrev.destination.rightOf(e):
                parents = [e.triangle]
                boundary_edge = e
            else:
                parents = [e.triangle, e.sym.triangle]
                e = e.oPrev
                self.deleteEdge(e.oNext)
        else:
            parents = [e.triangle]
            
        # Add point to triangulation
        self.vertexDict[self.nextVertexID] = v
        self.nextVertexID += 1
        
        # Create first spoke from origin of base to new site
        cdef Edge spoke = makeEdge(e.origin, v)
        self.addEdge(spoke)
        
        splice(spoke, e)
        cdef Edge startingSpoke = spoke
    
        # Create second spoke from destination of base to new site
        spoke = connect(e, spoke.sym)
        self.addEdge(spoke)
        
        e = spoke.oPrev
        while e.lNext != startingSpoke:
            spoke = connect(e, spoke.sym)
            self.addEdge(spoke)
            e = spoke.oPrev
        
        if boundary_edge is not None:
            # We might be deleting the base edge of the mesh, so use an edge
            # that is guaranteed to survive
            self.base = e
            self.deleteEdge(boundary_edge)
    
        # Create (potentially ephemeral) triangles for all the spokes and assign
        # them as children to the parent triangles
        cdef Edge currentSpoke = startingSpoke
        while True:
            currentSpoke = currentSpoke.dNext
            if currentSpoke.oNext.destination.leftOf(currentSpoke):
                child = Triangle(currentSpoke)
                created_triangles.append(child)
                for parent in parents:
                    parent.children.append(child)
                    
            if currentSpoke == startingSpoke:
                break
        
        for parent in parents:
            parent.anchor = None
        deleted_triangles.extend(parents)
            
    
        while True:
            t = e.oPrev
            if t.destination.rightOf(e) and v.inCircle(e.origin,
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
                    
                e = e.oPrev
            elif e.oNext == startingSpoke:
                break
            else:
                e = e.oNext.lPrev
        
        return created_triangles, deleted_triangles

    def scan_triangle(self, Triangle t):
        cdef Vertex v0, v1, v2
        v0, v1, v2 = t.vertices
        cdef int y_start, y_end
        cdef float x_a, x_b, dx0, dx1
        
        
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
    
    def scan_triangle_line(self, Triangle t, int y, float x_a, x_b):
        cdef int x_start = int(ceil(min(x_a, x_b)))
        cdef int x_end   = int(floor(max(x_a, x_b)))
        
        for x in range(x_start, x_end + 1):
            z_map = self.Hmap[y, x]
            error = abs(z_map - t.interpolate(x, y))
            if error > t.candidate_error:
                t.candidate_error = error
                t.candidate.pos = (x, y, z_map)
    
    def insert_next(self):
        cdef float error
        cdef Vertex candidate
        cdef Triangle triangle
        cdef list new, deleted
        
        error, (candidate, triangle) = self.heap.pop()
        new, deleted = self.insertSite(candidate)
    
        for triangle in new:
                self.scan_triangle(triangle)
                triangle.ID = self.heap.insert(triangle.candidate_error,
                                               (triangle.candidate, triangle))
    
        for triangle in deleted:
            if self.heap.contains(triangle.ID):
                self.heap.delete(triangle.ID)