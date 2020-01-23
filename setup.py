# -*- coding: utf-8 -*-
"""rpsimlib: Red Points image similarity library.

..note:
    "python setup.py test" invokes pytest on the package.
    Check test configuration in setup.cfg.

"""
from setuptools import find_packages
from setuptools import setup

setup(
    packages=find_packages(
        exclude=[
            'bak',
            'data',
            'examples',
            'scripts',
            'tests',
        ],
    ),
    pbr=True,
)
