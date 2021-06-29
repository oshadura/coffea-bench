#!/usr/bin/env python

import sys
import os.path

from setuptools import find_packages
from setuptools import setup



setup(name='coffea-bench',
      version='0.1',
      description='Benchmarks for Coffea',
      author='Oksana Shadura',
      author_email='oksana.shadura@cern.ch',
      url='https://github.com/oshadura/coffea-bench',
      install_requires=['coffea', 'dask', 'distributed', 'dask_jobqueue'],
      tests_require=['nbmake','pytest','pytest-benchmark','pytest-csv','pytest-benchmark[histogram]'],
      setup_requires=['pytest-runner'],
      test_suite = "tests",
)
