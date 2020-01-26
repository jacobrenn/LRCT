import distutils.core
from LRCT import __version__

distutils.core.setup(
    name='LRCT',
    version=__version__,
    packages=['LRCT'],
    author='Jacob Renn',
    author_email='jwrenn4@outlook.com',
    description='Python package for implementing Linear Regression Classification Trees',
    long_description=open('README.md').read(),
    requires=['scikit-learn', 'pandas', 'numpy']
)
