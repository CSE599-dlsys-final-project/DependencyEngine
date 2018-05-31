from __future__ import print_function

import os
import time

import numpy as np
import tvm

from dlsys import autodiff, tvm_op, dependency_engine
ad = autodiff

def test_heavy_data_reuse():
    # create context object
    tgt = "llvm"
    tgt_host = "llvm"
    executor_ctx = tvm.context(tgt, 0)
    print_loss_val_each_epoch = True
    num_epochs = 10

    print("=== Build computation graph...")

    A = ad.Variable(name="A")
    B = ad.Variable(name="B")
    C = ad.Variable(name="C")
    D = ad.Variable(name="D")
    E = ad.Variable(name="E")
    F = ad.Variable(name="F")
    G = ad.Variable(name="G")
    H = ad.Variable(name="H")

    z1 = ad.matmul_op(A, B)
    z2 = ad.matmul_op(A, C)
    z3 = ad.matmul_op(A, D)
    z4 = ad.matmul_op(A, E)
    z5 = ad.matmul_op(A, F)
    z6 = ad.matmul_op(A, G)

    y1 = z1 + A
    y2 = z2 + A
    y3 = z3 + A
    y4 = z4 + A
    y5 = z5 + A
    y6 = z6 + A

    executor = ad.Executor([y1, y2, y3, y4, y5, y6], ctx=executor_ctx)

    dim1 = (1000, 1000)
    dim2 = (1000, 1000)

    a = tvm.nd.array(np.ones(dim1, dtype=np.float32), ctx=executor_ctx)
    b = tvm.nd.array(np.ones(dim2, dtype=np.float32), ctx=executor_ctx)
    c = tvm.nd.array(np.ones(dim2, dtype=np.float32), ctx=executor_ctx)
    d = tvm.nd.array(np.ones(dim2, dtype=np.float32), ctx=executor_ctx)
    e = tvm.nd.array(np.ones(dim2, dtype=np.float32), ctx=executor_ctx)
    f = tvm.nd.array(np.ones(dim2, dtype=np.float32), ctx=executor_ctx)
    g = tvm.nd.array(np.ones(dim2, dtype=np.float32), ctx=executor_ctx)

    print("Starting...")
    time_measurements = []
    start_time = time.time()
    for i in range(num_epochs):
        y1, y2, y3, y4, y5, y6 = executor.run(feed_dict = {
            A: a,
            B: b,
            C: c,
            D: d,
            E: e,
            F: f,
            G: g
        })
        print(str(i) + " epoch(s) done")
        time_measurements.append(time.time() - start_time)
    print("Average Time per Training Epoch = %f s" % np.mean(time_measurements))


test_heavy_data_reuse()
