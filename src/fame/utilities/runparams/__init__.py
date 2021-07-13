#!/usr/bin/python

import os
import glob

tests = []

for name in os.listdir(os.path.dirname(__file__)):
    if name.endswith(".py") and name != '__init__.py':
        module = name[:-3]
        tests.append(module)
        __import__('%s.%s' % (__name__, module))

__all__ = []
