# distutils: language=c++

from libcpp.string cimport string

cdef extern from "DependencyEngine.hpp":
    cdef cppclass DependencyEngine:
        string getTen()

cdef class DependencyQueue:
    cdef DependencyEngine engine

    def getTen(self):
        return self.engine.getTen()
