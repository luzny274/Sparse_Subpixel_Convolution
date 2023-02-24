
from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from distutils import sysconfig


import shutil
import os
import numpy

if os.name == 'nt':
    setup(
        ext_modules = cythonize(Extension(
            "Sparse_Subpixel_Convolution",
            ["sparse_conv.pyx"],
            language="c++",
            extra_compile_args=["/Ox", "/openmp:experimental", "/std:c++17"],
            include_dirs=[numpy.get_include()]
        ), compiler_directives={'language_level' : "3"})
    )
else:
    setup(
        ext_modules = cythonize(Extension(
            "Sparse_Subpixel_Convolution",
            ["sparse_conv.pyx"],
            language="c++",
            extra_compile_args=["-O4", "-fopenmp", "-std=c++17"],
            include_dirs=[numpy.get_include()]
        ), compiler_directives={'language_level' : "3"})
    )


dir_name = "./"
test = os.listdir(dir_name)

for item in test:
    if item.endswith(".pyd") or item.endswith(".so"):
        shutil.copy(item, "../")
