import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import src.tests.algebra.function as tests_functions
import src.tests.algebra.generator as tests_generator
import src.tests.algebra.iterator as tests_iterator
import src.tests.algebra.linear as tests_linear
import src.tests.algebra.polynomial as tests_polynomial

def test_all():
    tests_functions.test_all()
    tests_generator.test_all()
    tests_iterator.test_all()
    tests_linear.test_all()
    tests_polynomial.test_all()


if __name__ == '__main__':
    test_all()