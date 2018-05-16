test_dependency_engine:
	python tests/test_dependency_engine.py

export:
	PYTHONPATH="${PYTHONPATH}:/Users/macbook/git/DependencyEngine/python"

clean:
	rm -rf ./python/dlsys/*.pyc
	rm -rf ./tests/*.pyc
