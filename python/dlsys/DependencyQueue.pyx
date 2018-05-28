# distutils: language=c++

from contextlib import contextmanager

from libcpp.unordered_set cimport unordered_set

cdef extern from "DependencyEngine.hpp":
    ctypedef void (*callback)(void *user_args)

    cdef cppclass DependencyEngine:
        void push(callback execFunc, void* args,
            unordered_set[long] readTags, unordered_set[long] mutateTags)

        long newVariable()

        void start()
        void stop()

cdef void engine_callback(void* args):
    f_node_to_val_map, f_node, f_node_val, func = (<object>args)
    input_vals = [f_node_to_val_map[n] for n in f_node.inputs]
    f_node.op.compute(f_node, input_vals, f_node_val, func)
    f_node_to_val_map[f_node] = f_node_val
    # print("Done executing engine_callback." + str((f_node_to_val_map, f_node, f_node_val, func)))

cdef class DependencyQueue:
    cdef DependencyEngine engine

    def push(self, args, readTags, mutateTags):
        self.engine.push(engine_callback, <void*>args, readTags, mutateTags)

    def new_variable(self):
        return self.engine.newVariable()

    def start(self):
        self.engine.start()

    def stop(self):
        self.engine.stop()

    @contextmanager
    def threaded_executor(self):
        self.start()
        yield
        self.stop()
