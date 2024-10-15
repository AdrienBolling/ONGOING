# setup.py
from distutils.command.install import install

from setuptools import setup, find_packages

setup(
    name='ongoing',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'jax',
        'flax',
        'plotly'
        ],
    description='ONGOING method for knowledge modelling',
    author='Adrien Bolling',
    author_email='bolling.adrien@gmail.com',
    url='https://github.com/AdrienBolling/ongoing',
)
