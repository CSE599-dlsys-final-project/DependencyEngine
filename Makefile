install:
	pip3 install -e ./python/ --user

test_all: test_dependency_engine test_threading test_reuse

test_dependency_engine:
	python3 tests/test_dependency_engine.py

test_threading:
	python3 tests/test_threading.py

test_reuse:
	python3 tests/test_heavy_data_reuse.py

export:
	PYTHONPATH="${PYTHONPATH}:/Users/macbook/git/DependencyEngine/python"

clean:
	rm -rf ./python/dlsys/*.pyc
	rm -rf ./tests/*.pyc
	rm -rf ./python/build
