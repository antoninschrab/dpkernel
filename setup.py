#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

dist = setup(
    name="dpkernel",
    version="1.0.0",
    description="Differentially Private Permutation Tests: Applications to Kernel Methods",
    author="Antonin Schrab",
    author_email="a.lastname@ucl.ac.uk",
    license="MIT License",
    packages=["dpkernel", ],
    install_requires=["jax", "jaxlib"],
    python_requires=">=3.9",
)
