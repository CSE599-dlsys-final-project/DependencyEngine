test_logreg:
	python tests/mnist_dlsys.py -l -m logreg

test_mlp:
	python tests/mnist_dlsys.py -l -m mlp

test_op:
	nosetests -v tests/test_tvm_op.py

export:
	PYTHONPATH="${PYTHONPATH}:/Users/macbook/git/assignment2-2018/python"

clean:
	rm -rf ./python/dlsys/*.pyc
	rm -rf ./tests/*.pyc
