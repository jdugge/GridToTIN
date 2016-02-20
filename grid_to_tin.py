#!/usr/bin/python

import sys

import numpy as np

import triangulation as tri
import heap
import quadedge as qe

#import shapely.geometry as geo
import rasterio


def write_ply(filename, coordinates, triangles, binary=True):
    template = "ply\n"
    if binary:
        template += "format binary_" + sys.byteorder + "_endian 1.0\n"
    else:
        template += "format ascii 1.0\n"
    template += """element vertex {nvertices:n}
property float x
property float y
property float z
element face {nfaces:n}
property list int int vertex_index
end_header
"""

    context = {
     "nvertices": len(coordinates),
     "nfaces": len(triangles)
    }

    if binary:
        with open(filename,'wb') as outfile:
            outfile.write(template.format(**context))
            coordinates = np.array(coordinates, dtype="float32")
            coordinates.tofile(outfile)

            triangles = np.hstack((np.ones([len(triangles),1], dtype="int") * 3,
                triangles))
            triangles = np.array(triangles, dtype="int32")
            triangles.tofile(outfile)
    else:
        with open(filename,'w') as outfile:
            outfile.write(template.format(**context))
            np.savetxt(outfile, coordinates, fmt="%.3f")
            np.savetxt(outfile, triangles, fmt="3 %i %i %i")

def write_obj(filename, coordinates, triangles, binary=False):
    texture_coordinates = coordinates[:,:2].copy()
    texture_coordinates -= texture_coordinates.min(axis=0)
    texture_coordinates /= texture_coordinates.ptp(axis=0)

    template = """mtllib material.mtl\n
usemtl material0\n
\n"""
    if binary:
        template += "format binary_" + sys.byteorder + "_endian 1.0\n"
    else:
        pass
        #template += "format ascii 1.0\n"
    template += """"""

    context = {}

    if binary:
        with open(filename, 'wb') as outfile:
            outfile.write(template.format(**context))
            coordinates = np.array(coordinates, dtype="float32")
            coordinates[:, 2] = coordinates[:,2] * 100
            coordinates.tofile(outfile)

            triangles = np.hstack((np.ones([len(triangles),1], dtype="int") * 3,
                triangles))
            triangles = np.array(triangles, dtype="int32")
            triangles.tofile(outfile)
    else:
        with open(filename, 'wb') as outfile:
            outfile.write(bytes(template.format(**context), 'UTF-8'))
            np.savetxt(outfile, coordinates, fmt="v %.3f %.3f %.3f")
            np.savetxt(outfile, texture_coordinates, fmt="vt %.3f %.3f")
            np.savetxt(outfile,
                       np.dstack([triangles, triangles]).reshape(-1,6) + 1,
                       fmt="f %i/%i/ %i/%i/ %i/%i/")

def main(argv):
    inputfile = argv[0]
    nvertices = int(argv[1])
    outputfile = argv[2]
    z_scale = float(argv[3])

    with rasterio.open(inputfile) as file:
        raster = file.read().astype(float).squeeze()
        affine = file.affine

    t = tri.Triangulation(raster)
    for i in range(nvertices):
        t.insert_next()

    vertices = np.array([affine * v.pos[:2] + tuple([v.pos[2] * z_scale]) for v in t.vertices])
    triangles = [[ t.vertices.index(vertex) for vertex in triangle.vertices][::-1]
                for triangle in t.triangles ]

    write_obj(outputfile, vertices, triangles, binary=False)

if __name__ == "__main__":
    main(sys.argv[1:])
