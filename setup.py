import os
import glob
from distutils.core import setup

scripts=['great3-run','great3-mean-shear','great3-calc-skynoise',
         'great3-make-condor','great3-make-wq','great3-combine-ranges',
         'great3-calc-q']

scripts=['great-des-average',
         'great-des-combine',
         'great-des-combine-averaged',
         'great-des-fit',
         'great-des-gen-condor',
         'great-des-gen-wq']

scripts=[os.path.join('bin',s) for s in scripts]

setup(name="great_des", 
      version="0.1.0",
      description="Run on greatdes",
      license = "GPL",
      author="Erin Scott Sheldon",
      author_email="erin.sheldon@gmail.com",
      scripts=scripts,
      packages=['great_des'])
