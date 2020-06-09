"""
Setup script for a clean pip install.

*************************************
"""
from setuptools import setup, find_packages

setup(name='slideminer',
      version='0.1',
      description="procedures for weakly supervised knowledge extraction\
                   in whole slide images cohorts.",
      author='Arnaud Abreu',
      author_email='arnaud.abreu.p@gmail.com',
      packages=find_packages(),
      zip_safe=False,
      install_requires=[],
      include_package_data=True)
