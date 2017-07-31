import os
import sys
import shutil
from subprocess import call
from setuptools import setup
from warnings import warn

if sys.version_info.major != 3:
    raise RuntimeError('SCAnalysis requires Python 3')

# install phenograph if pip3 is installed
if shutil.which('pip3'):
    call(['pip3', 'install', 'git+https://github.com/jacoblevine/phenograph.git'])
    call(['pip3', 'install', 'git+https://github.com/pkathail/GraphDiffusion.git'])
# install GraphDiffusion if pip3 is installed ^



setup(name='SCAnalysis',
      version='0.0',
      description='Single Cell Analysis',
      url='https://github.com/helenjin/scanalysis',
      author='',
      author_email='',
      package_dir={'': 'src'},
      packages=['scanalysis', 'scanalysis.io', 'scanalysis.tools', 'scanalysis.tools.wb', 'scanalysis.tools.pr', 'scanalysis.utils', 'scanalysis.plots'],
      install_requires=[
          'numpy>=1.12.0',
          'pandas>=0.19.2',
          'scipy>=0.18.1',
          'matplotlib',
          'bhtsne',
          'matplotlib>=1.5.1',
          'seaborn>=0.7.1',
          'sklearn',
          'networkx>=1.11',
          'fcsparser>=0.1.2',
          'statsmodels>=0.8.0',
          'GraphDiffusion'
      ],
     scripts=['src/scanalysis/sca_gui.py'],
      )


# get location of setup.py
setup_dir = os.path.dirname(os.path.realpath(__file__))

# install GSEA, diffusion components
tools_dir = os.path.expanduser('~/.scanalysis/tools')
if os.path.isdir(tools_dir):
    shutil.rmtree(tools_dir)
shutil.copytree(setup_dir + '/tools/', tools_dir)
shutil.unpack_archive(tools_dir + '/mouse_gene_sets.tar.gz', tools_dir)
shutil.unpack_archive(tools_dir + '/human_gene_sets.tar.gz', tools_dir)

# Copy test data
data_dir = os.path.expanduser('~/.scanalysis/data')
if os.path.isdir(data_dir):
    shutil.rmtree(data_dir)
shutil.copytree(setup_dir + '/data/', data_dir)

# Create directory for GSEA reports
os.makedirs( os.path.expanduser('~/.scanalysis/gsea/'), exist_ok=True )
