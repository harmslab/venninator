__description__ = \
"""
estimate overlaps between sets with false positive and false negative rates
"""

__version__ = 0.1

import sys 
if sys.version_info[0] < 3:
    sys.exit('Sorry, Python < 3.x is not supported')

from setuptools import setup, find_packages
import numpy

setup(name="venninator",
      packages=find_packages(),
      version=__version__,
      description="estimate overlaps between sets with false positive and false negative rates",
      long_description=__description__,
      author='Michael J. Harms',
      author_email='harmsm@gmail.com',
      url='https://github.com/harmslab/venninator',
      download_url="https://github.com/harmslab/venninator/archive/{}.tar.gz".format(__version__),
      install_requires=["numpy","scipy","emcee","corner"],
      package_data={},
      zip_safe=False,
      classifiers=['Programming Language :: Python']
      )
