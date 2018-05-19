from __future__ import print_function

import numpy as np
import tvm
from dlsys import autodiff, tvm_op, dependency_engine


tgt_host="llvm"
tgt="llvm"
dtype = "float32"
ctx = tvm.context(tgt, 0)

def test_matrix_elementwise_add_naive():
    print("******")
    print("Testing naive elemwise add")
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

    ### running the fake executor as we push in instructions
    ### where we just simulate it by as if it keeps running
    for _ in range(5):
        engine.naive_executor()
    elemwise_add = tvm_op.make_elemwise_add(shape, tgt, tgt_host, "elem_add")
    # push first instruction (tvm instruction)
    engine.push(lambda: elemwise_add(arr_x, arr_y, arr_z), [x_tag, y_tag], [z_tag])
    for _ in range(5):
        engine.naive_executor()

    # test if we got it right
    # since we not running on threads, no need for blocking
    z = arr_z.asnumpy()
    np.testing.assert_allclose(x + y, z, rtol=1e-5)

    # push second instruction (print intruction)
    engine.push(lambda: print(arr_z.asnumpy()), [z_tag], [])
    for _ in range(5):
        engine.naive_executor()


test_matrix_elementwise_add_naive()
