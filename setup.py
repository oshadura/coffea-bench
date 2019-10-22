#!/usr/bin/env python

import sys
import os.path

from setuptools import find_packages
from setuptools import setup

setup(name='coffea-bench',
      version='0.1',
      description='Benchmarks for Coffea andf Laurelin',
      author='Oksana Shadura',
      author_email='oksana.shadura@cern.ch',
      url="https://github.com/oshadura/coffea-bench",
      install_requires=["coffea", "pyspark"],
      tests_require=["pytest","pytest-benchmark"],
      setup_requires=["pytest-runner"],
      packages='find_packages(exclude=["tests"])',
)