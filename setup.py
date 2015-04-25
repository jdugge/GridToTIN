from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from numpy.distutils.misc_util import get_numpy_include_dirs


numpy_dirs = get_numpy_include_dirs()

setup(
    ext_modules = cythonize(["quadedge.pyx"], include_dirs=numpy_dirs)
)
