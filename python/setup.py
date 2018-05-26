#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension

from Cython.Build import cythonize

compile_args = ['-g', '-std=c++11', '-stdlib=libc++']

cppExtension = Extension(
    "dependencyengine",
    ["dlsys/*.pyx", "dlsys/cpp/DependencyEngine.cpp"],
    include_dirs=[".", "dlsys/cpp"],
    extra_compile_args=compile_args
)

setup(
    name='DlSysDependencyEngine',
    version='0.9',
    description='Dependency engine for computation graph executor',
    ext_modules = cythonize([cppExtension])
)
