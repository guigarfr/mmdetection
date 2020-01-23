#!/usr/bin/env python
"""
Script to run tests in bamboo plan.

This scripts ignores the exit code of pytest execution
so the bamboo plan continues to next step evaluating
the junit xml obtained from the test execution.
"""
import sys

import pytest


if __name__ == '__main__':
    pytest.main(sys.argv[1:])
