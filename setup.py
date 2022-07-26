from setuptools import setup
from LRCT import __version__

with open('requirements.txt', 'r') as f:
    requirements = [line for line in f.read().splitlines() if line != '']

setup(
    name='LRCT',
    version=__version__,
    packages=['LRCT'],
    author='Jacob Renn',
    author_email='jwrenn@captechu.edu',
    description='Python package for implementing Linear Regression Classification Trees',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license='Apache 2.0',
    install_requires=requirements
)
