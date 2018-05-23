from __future__ import print_function

import argparse
import six.moves.cPickle as pickle
import gzip
import os
import time

import numpy as np
import tvm

from dlsys import autodiff, tvm_op, dependency_engine
ad = autodiff
from mnist_dlsys import load_mnist_data, convert_to_one_hot


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

def test_mnist_logreg():
    tgt = "llvm"
    tgt_host = "llvm"

    # create context object
    executor_ctx = tvm.context(tgt, 0)
    print_loss_val_each_epoch = True
    num_epochs = 10

    print("=== Build logistic regression model...")

    # recover tgt, tgt_host info from tvm.context
    if executor_ctx == tvm.cpu(0):
        tgt = "llvm"
        tgt_host = "llvm"
    else:
        assert False, "non-CPU context not yet supported"

    W1 = ad.Variable(name="W1")
    b1 = ad.Variable(name="b1")
    X = ad.Variable(name="X")
    y_ = ad.Variable(name="y_")

    z1 = ad.matmul_op(X, W1)
    y = z1 + ad.broadcastto_op(b1, z1)

    loss = ad.softmaxcrossentropy_op(y, y_)

    grad_W1, grad_b1 = ad.gradients(loss, [W1, b1])
    executor = ad.Executor([loss, grad_W1, grad_b1, y], ctx=executor_ctx)

    # Read input data
    datasets = load_mnist_data("mnist.pkl.gz")
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # Set up minibatch
    batch_size = 1000
    n_train_batches = train_set_x.shape[0] // batch_size
    n_valid_batches = valid_set_x.shape[0] // batch_size

    print("Start training loop...")

    # Initialize parameters
    W1_val = np.zeros((784, 10), dtype=np.float32)
    b1_val = np.zeros((10), dtype=np.float32)
    X_val = np.empty(shape=(batch_size, 784), dtype=np.float32)
    y_val = np.empty(shape=(batch_size, 10), dtype=np.float32)
    valid_X_val = np.empty(shape=(batch_size, 784), dtype=np.float32)
    valid_y_val = np.empty(shape=(batch_size, 10), dtype=np.float32)

    # wrap them under tvm.nd.array
    W1_val = tvm.nd.array(W1_val, ctx=executor_ctx)
    b1_val = tvm.nd.array(b1_val, ctx=executor_ctx)
    X_val = tvm.nd.array(X_val, ctx=executor_ctx)
    y_val = tvm.nd.array(y_val, ctx=executor_ctx)
    valid_X_val = tvm.nd.array(valid_X_val, ctx=executor_ctx)
    valid_y_val = tvm.nd.array(valid_y_val, ctx=executor_ctx)

    # training loop
    lr = 1e-3
    # JIT compile sgd update ops
    W1_sgd_update_func = tvm_op.make_sgd_update(
        W1_val.shape, lr, tgt, tgt_host, "W1_sgd_update")
    b1_sgd_update_func = tvm_op.make_sgd_update(
        b1_val.shape, lr, tgt, tgt_host, "b1_sgd_update")
    time_measurements = []
    for i in range(num_epochs):
        print("epoch %d" % i)
        start_time = time.time()
        for minibatch_index in range(n_train_batches):
            minibatch_start = minibatch_index * batch_size
            minibatch_end = (minibatch_index + 1) * batch_size
            X_val.copyfrom(train_set_x[minibatch_start:minibatch_end])
            y_val.copyfrom(
                convert_to_one_hot(train_set_y[minibatch_start:minibatch_end]))
            loss_val, grad_W1_val, grad_b1_val, _ = executor.run_with_dependency_engine(
                feed_dict = {X: X_val, y_: y_val, W1: W1_val, b1: b1_val})
            # SGD update
            # W1_val = W1_val - lr * grad_W1_val
            # b1_val = b1_val - lr * grad_b1_val
            W1_sgd_update_func(W1_val, grad_W1_val, W1_val)
            b1_sgd_update_func(b1_val, grad_b1_val, b1_val)
        time_measurements.append(time.time() - start_time)
        if print_loss_val_each_epoch:
            print("loss = %f; Time taken this epoch = %f s"
                % (np.asscalar(loss_val.asnumpy()), time_measurements[-1]))

    correct_predictions = []
    for minibatch_index in range(n_valid_batches):
        minibatch_start = minibatch_index * batch_size
        minibatch_end = (minibatch_index + 1) * batch_size
        valid_X_val.copyfrom(valid_set_x[minibatch_start:minibatch_end])
        valid_y_val.copyfrom(
            convert_to_one_hot(valid_set_y[minibatch_start:minibatch_end]))
        _, _, _, valid_y_predicted = executor.run_with_dependency_engine(
            feed_dict={
                        X: valid_X_val,
                        y_: valid_y_val,
                        W1: W1_val,
                        b1: b1_val},
            convert_to_numpy_ret_vals=True)
        correct_prediction = np.equal(
            np.argmax(valid_y_val.asnumpy(), 1),
            np.argmax(valid_y_predicted, 1)).astype(np.float)
        correct_predictions.extend(correct_prediction)
    accuracy = np.mean(correct_predictions)
    # validation set accuracy=0.928200
    print("Validation set accuracy = %f" % accuracy)
    print("Average Time per Training Epoch = %f s" % np.mean(time_measurements))



test_matrix_elementwise_add_naive()
test_mnist_logreg()
