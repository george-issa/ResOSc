from setuptools import setup, find_packages

setup(
    name='ResOSc',
    version='0.1.0',
    description='Resonant Oscillator Simulator for Coupled Systems',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.21',
        'scipy>=1.7',
        'matplotlib>=3.5',
    ],
)
