from __future__ import print_function

import numpy as np
import tvm
from dlsys import autodiff, tvm_op, dependency_engine


tgt_host="llvm"
tgt="llvm"
dtype = "float32"
ctx = tvm.context(tgt, 0)

def test_matrix_elementwise_add_threaded():
    print("******")
    print("Testing threaded elemwise add")
    ### preparing data
    shape = (500, 200)
    x = np.random.uniform(0, 10, size=shape).astype(dtype)
    y = np.random.uniform(0, 10, size=shape).astype(dtype)
    z = np.zeros(shape).astype(dtype)
    # put into tvm array
    arr_x = tvm.nd.array(x, ctx=ctx)
    arr_y = tvm.nd.array(y, ctx=ctx)
    arr_z = tvm.nd.array(z, ctx=ctx)

    ### prepare engine
    engine = dependency_engine.DependencyEngine()
    # resource tags
    x_tag = engine.new_variable("X")
    y_tag = engine.new_variable("Y")
    z_tag = engine.new_variable("Z")

    # start execution engine!
    engine.start_threaded_executor()
    ### running the fake executor as we push in instructions
    ### where we just simulate it by as if it keeps running
    elemwise_add = tvm_op.make_elemwise_add(shape, tgt, tgt_host, "elem_add")
    # push first instruction (tvm instruction)
    engine.push(lambda: elemwise_add(arr_x, arr_y, arr_z), [x_tag, y_tag], [z_tag])
    # push second instruction (print intruction)
    engine.push(lambda: print(arr_z.asnumpy()), [z_tag], [])
    # blocking call
    engine.stop_threaded_executor()
    # test if we got it right
    # since we not running on threads, no need for blocking
    z = arr_z.asnumpy()
    np.testing.assert_allclose(x + y, z, rtol=1e-5)
    print("All done!")

# tests if the intrcutions will be performed in the desired order
def test_threaded_dependency():
    print("******")
    print("Testing threaded dependency")
    ### prepare engine
    engine = dependency_engine.DependencyEngine()
    # resource tags
    x_tag = engine.new_variable("X")
    y_tag = engine.new_variable("Y")

    # start execution engine!
    engine.start_threaded_executor()
    # Notice the push order is not the same as execution order.
    # Although "Reading x" is pushed last, its execution is non-deterministic;
    # we can only gaurentee that "Reading x, modifying y" comes after "Modifying y"
    engine.push(lambda: print("Modifying y"), [], [y_tag])
    engine.push(lambda: print("Reading x, modifying y"), [x_tag], [y_tag])
    engine.push(lambda: print("Reading x"), [x_tag], [])
    # blocking call
    engine.stop_threaded_executor()
    print("All done!")


def test_known_bug_case():
    print("******")
    print("Testing threaded dependency")
    ### prepare engine
    engine = dependency_engine.DependencyEngine()
    # resource tags
    a_tag = engine.new_variable("A")
    b_tag = engine.new_variable("B")
    c_tag = engine.new_variable("C")
    d_tag = engine.new_variable("D")
    z_tag = engine.new_variable("Z")

    class MyInt(object):
        def __init__(self, num):
            self.num = num

        def __str__(self):
            return str(self.num)

    A = MyInt(1)
    B = MyInt(1)
    C = MyInt(1)
    D = MyInt(1)
    Z = MyInt(1)

    # start execution engine!
    engine.start_threaded_executor()
    # Notice the push order is not the same as execution order.
    # Although "Reading x" is pushed last, its execution is non-deterministic;
    # we can only gaurentee that "Reading x, modifying y" comes after "Modifying y"
    def fn1():
        print("1!")
        A.num=B.num+C.num
    def fn2():
        print("2!")
        D.num=A.num+Z.num
    def fn3():
        print("3!")
        C.num=D.num

    for _i in range(1, 4):
        def fn(i):
            if i == 1:
                fn1()
            elif i == 2:
                fn2()
            else:
                fn3()
        if _i == 1:
            engine.push(lambda: fn(_i), [b_tag, c_tag], [a_tag])
        elif _i == 2:
            engine.push(lambda: fn(_i), [a_tag, z_tag], [d_tag])
        else:
            engine.push(lambda: fn(_i), [d_tag], [c_tag])

    # blocking call
    engine.stop_threaded_executor()


    print("A: "+ str(A))
    print("B: "+ str(B))
    print("C: "+ str(C))
    print("D: "+ str(D))
    print("Z: "+ str(Z))

test_known_bug_case()
test_matrix_elementwise_add_threaded()
test_threaded_dependency()
