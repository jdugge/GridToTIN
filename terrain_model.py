import scipy.spatial as ss
import scipy.interpolate as si
import scipy.optimize as so
import numpy as np
import rasterio
import triangle

import matplotlib.tri as mpltri
import matplotlib.pyplot as plt


def norm(a, order=2):
    return np.linalg.norm(a.flatten(), ord=order)/a.size**(1/order)


class TerrainModel:
    def __init__(self, dem):
        with rasterio.drivers():
            with rasterio.open(dem) as src:
                raw_data = src.read()
                self.dem = np.array(raw_data.squeeze(), dtype=float)
        self.xx, self.yy = np.indices(self.dem.shape)

        self.points = np.array([[                  0,                   0],
                                [self.dem.shape[1]-1,                   0],
                                [self.dem.shape[1]-1, self.dem.shape[0]-1],
                                [                  0, self.dem.shape[0]-1]])
        self.point_elevations = self.dem_elevation(*self.points.T)

        self.triangulation_dirty = True
        self._triangulation = self.triangulation()

        self.interpolation_map_dirty = True
        self._interpolation_map = self.interpolation_map()

        self.improvement_map = np.zeros_like(self.dem)
        self.improvement_map[:] = np.nan

    def dem_elevation(self, x, y):
        return self.dem[y, x]

    def insert_point(self, x, y):
        self.points = np.append(self.points, [[x,y]], axis=0)
        self.point_elevations = np.append(self.point_elevations,
                                          [self.dem_elevation(x, y)], axis=0)
        self.triangulation_dirty = True
        self.interpolation_map_dirty = True

    def triangulation(self, recalculate=False):
        if recalculate or self.triangulation_dirty:
            self._triangulation = triangle.delaunay(self.points)
            self._triangulation = mpltri.Triangulation(*self.points.T,
                                                       triangles=self._triangulation)

        self.triangulation_dirty = False
        return self._triangulation

    def interpolation_map(self, recalculate=False):
        if recalculate or self.interpolation_map_dirty:
            interpolator = mpltri.LinearTriInterpolator(self.triangulation(),
                                                        self.point_elevations)
            self._interpolation_map = interpolator(self.yy, self.xx)

        self.interpolation_map_dirty = False
        return self._interpolation_map

    def error_map(self):
        return self.interpolation_map() - self.dem

    def plot_triangulation(self):
        error_map = self.error_map()
        max_error = np.max(np.abs(error_map))

        min_elevation = np.min(self.dem)
        max_elevation = np.max(self.dem)

        fig, ax = plt.subplots(1,3, figsize=(15, 6))
        ax[0].imshow(self.dem,
                     origin='top',
                     cmap='viridis',
                     vmin=min_elevation,
                     vmax=max_elevation)
        ax[0].triplot(self.points[:,0],
                      self.points[:,1],
                      self.triangulation().triangles,
                      color='red', linewidth=2)
        ax[0].margins(0)

        ax[1].imshow(self.interpolation_map(),
                     origin='top',
                     cmap='viridis',
                     vmin=min_elevation,
                     vmax=max_elevation)
        ax[1].triplot(self.points[:,0],
                      self.points[:,1],
                      self.triangulation().triangles,
                      color='red', linewidth=2)
        ax[1].margins(0)

        ax[2].imshow(error_map, origin='top',
                     cmap='RdBu',
                     vmin=-max_error,
                     vmax=max_error)
        ax[2].triplot(self.points[:,0],
                      self.points[:,1],
                      self.triangulation().triangles,
                      color='red', linewidth=2)
        ax[2].margins(0)

    def test_point(self, p, error=None, interpolation=None, error_norm_order=2, update_improvement_map=False):
        # Calculate old error
        error_old = norm(self.interpolation_map() - self.dem, error_norm_order)

        # Append the new coordinates
        p = np.round(p).astype(int)
        points = np.vstack([self.points, [p]])
        values = np.append(self.point_elevations, self.dem_elevation(p[0], p[1]))

        # Retriangulate
        tri_new = triangle.delaunay(points)
        tri_new = mpltri.Triangulation(*points.T, triangles=tri_new)

        # Reinterpolate
        interpolator = mpltri.LinearTriInterpolator(tri_new, values)
        interpolation_new = interpolator(self.yy, self.xx)

        # Calculate new error
        error_new = norm(interpolation_new - self.dem, error_norm_order)

        improvement = error_new - error_old

        if update_improvement_map:
            self.improvement_map[p[1], p[0]] = improvement

        return improvement

    def point_with_greatest_improvement(self, error_norm_order):
        self.improvement_map[:] = np.nan

        if error_norm_order == np.inf:
            opt = so.brute(self.test_point,
                                       [(0,self.dem.shape[1]-1),
                                        (0,self.dem.shape[0]-1)],
                                       Ns=25, args = (None, None, np.inf, True))
            x_new, y_new = np.round(opt).astype(int)
            improvement = np.nanmin(self.improvement_map)
        else:
            opt = so.differential_evolution(self.test_point,
                                                        args = [None, None, error_norm_order, True],
                                                        bounds = [(0,self.dem.shape[1]-1),
                                                                  (0,self.dem.shape[0]-1)],
                                                        popsize=20, tol=0.0001)
            x_new, y_new = np.round(opt.x).astype(int)
            improvement = opt.fun
        return (x_new, y_new), improvement

    def point_with_greatest_error(self):
        y, x = np.unravel_index(np.argmax(np.abs(self.error_map().flatten())), self.dem.shape)
        error = self.error_map()[y,x]
        return (x, y), error

    def full_improvement_map(self, error_norm_order=2):
        yy, xx = np.indices(self.dem.shape)
        self.improvement_map[:] = np.nan
        numpoints = len(self.dem.flatten())

        for i, (x, y) in enumerate(zip(xx.flatten(), yy.flatten())):
            percent_done = round((i+1)/numpoints*100, 1)
            print('{:>5}'.format(percent_done), "%: Testing point:", (x,y), end="")
            clear_output(wait=True)
            tm.test_point([x, y], error_norm_order=error_norm_order, update_improvement_map=True)