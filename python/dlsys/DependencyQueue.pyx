# distutils: language=c++

from libcpp.string cimport string
from libcpp.set cimport set

cdef extern from "DependencyEngine.hpp":
    cdef cppclass DependencyEngine:
        void push(long execFunc, set[long] readTags, set[long] mutateTags)

        long newVariable()

        void start()
        void stop()

cdef class DependencyQueue:
    cdef DependencyEngine engine

    def push(self, execFunc, readTags, mutateTags):
        self.engine.push(0, readTags, mutateTags)

    def new_variable(self):
        return self.engine.newVariable()

    def start(self):
        self.engine.start()

    def stop(self):
        self.engine.stop()
