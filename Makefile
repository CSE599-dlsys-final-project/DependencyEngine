test_all: test_dependency_engine test_threading

test_dependency_engine:
	python tests/test_dependency_engine.py

test_threading:
	python tests/test_threading.py

export:
	PYTHONPATH="${PYTHONPATH}:/Users/macbook/git/DependencyEngine/python"

clean:
	rm -rf ./python/dlsys/*.pyc
	rm -rf ./tests/*.pyc
