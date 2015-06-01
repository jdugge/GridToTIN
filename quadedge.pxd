import numpy as np
cimport numpy as np
np.import_array()

cdef class Edge:
    cdef public Edge next
    cdef public Edge _rot
    cdef public Vertex origin
    cdef public int flag
    cdef public int ID
    cdef public Triangle triangle
    
    cpdef asLineSegment(self)

cdef class Vertex:
    cdef public long x, y
    cdef public float z
    cdef public Edge parent
    
    cpdef inCircle(self, Vertex v0, Vertex v1, Vertex v2)
    cpdef inTriangle(self, Vertex v0, Vertex v1, Vertex v2)
    cpdef leftOf(self, Edge e)
    cpdef rightOf(self, Edge e)
    cpdef onEdge(self, Edge e)
    cpdef inTriangleSlow(self, Triangle tri)

cdef class Triangle:
    cdef public int ID
    cdef public Edge anchor
    cdef public list vertices
    cdef public list children
    cdef public int area
    
    cdef public float a, b, c # Plane equation parameters
    cdef public Vertex candidate
    cdef public float candidate_error
    
    cpdef reshape(self)
    cpdef calculate_plane_equation(self)
    cpdef interpolate(self, int x, int y)
    cpdef rasterize_triangle(self, np.ndarray[np.float_t, ndim=2] H)
    cdef rasterize_flat(self, Vertex v0, Vertex v1, Vertex v2, 
                        np.ndarray[np.float_t, ndim=2] H, include_last_row = ?,
                        float max_error = ?)

cpdef Edge splice(Edge a, Edge b)

cpdef Edge connect(Edge a, Edge b)

cpdef swap(Edge e)

cpdef makeTriangle(Vertex v0, Vertex v1, Vertex v2)

cpdef deleteEdge(Edge e)

cpdef Edge makeEdge(Vertex origin, Vertex destination)