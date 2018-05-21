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
    engine = dependency_engine.Dependency_Engine()
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
    engine = dependency_engine.Dependency_Engine()
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

test_matrix_elementwise_add_threaded()
test_threaded_dependency()