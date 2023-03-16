
from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from distutils import sysconfig


import shutil
import os
import numpy

path = os.path.dirname(__file__)
print(os.path.dirname(__file__))

sanitize = False

if not sanitize:
    if os.name == 'nt':
        setup(
            ext_modules = cythonize(Extension(
                "Sparse_Subpixel_Convolution",
                ["sparse_conv.pyx"],
                language="c++",
                extra_compile_args=["/Ox", "/openmp:experimental", "/std:c++17", "/MD"],
                # extra_link_args=["libhoard.lib"],
                include_dirs=[numpy.get_include()]
            ), compiler_directives={'language_level' : "3"})
        )
    else:
        setup(
            ext_modules = cythonize(Extension(
                "Sparse_Subpixel_Convolution",
                ["sparse_conv.pyx"],
                language="c++",
                extra_compile_args=["-O4", "-fopenmp", "-std=c++17", "-pthread"],
                extra_link_args = ['-fopenmp', "-pthread"],
                include_dirs=[numpy.get_include()]
            ), compiler_directives={'language_level' : "3"})
        )
else:
    # sanitizer = "-fsanitize=thread"
    sanitizer = "-fsanitize=address"
    if os.name == 'nt':
        setup(
            ext_modules = cythonize(Extension(
                "Sparse_Subpixel_Convolution",
                ["sparse_conv.pyx"],
                language="c++",
                extra_compile_args=["/Od", "/openmp:experimental", "/std:c++17", "/MD", sanitizer, "/DEBUG", "/Zi"],
                extra_link_args = ["/DEBUG"],
                include_dirs=[numpy.get_include()]
            ), compiler_directives={'language_level' : "3"})
        )
    else:
        setup(
            ext_modules = cythonize(Extension(
                "Sparse_Subpixel_Convolution",
                ["sparse_conv.pyx"],
                language="c++",
                extra_compile_args=["-O0", "-fopenmp", "-std=c++17", sanitizer, "-pthread"],
                extra_link_args = ['-fopenmp', sanitizer, "-pthread"],
                include_dirs=[numpy.get_include()]
            ), compiler_directives={'language_level' : "3"})
        )


dir_name = "./"
test = os.listdir(dir_name)

for item in test:
    if item.endswith(".pyd") or item.endswith(".so"):
        shutil.copy(item, "../")
