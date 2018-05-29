#!/usr/bin/env python

import glob

from setuptools import setup
from setuptools.extension import Extension

from Cython.Build import cythonize

cpp_source_files = glob.glob("dlsys/cpp/*.cpp")
cpp_depends = glob.glob("dlsys/cpp/*.hpp")

cppExtension = Extension(
    "dependencyengine",
    ["dlsys/*.pyx"] + cpp_source_files,
    include_dirs=[".", "dlsys/cpp"],
    depends=cpp_depends,
    extra_compile_args=['-g', '-std=c++14', '-stdlib=libc++']
)

setup(
    name='DlSysDependencyEngine',
    version='0.9',
    description='Dependency engine for computation graph executor',
    ext_modules=cythonize([cppExtension]),
    install_requires=[
        "cython~=0.28"
    ]
)
