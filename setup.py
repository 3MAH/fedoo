#!/usr/bin/env python

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(name='fedoo',
      version='1.0',
      description='Microstructure generation',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author="3MAH",
      author_email="set3mah@gmail.com",
      url='https://github.com/3MAH/fedoo',
      packages=find_packages(),
      python_requires=">=3.7",
)
