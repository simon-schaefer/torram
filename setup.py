from setuptools import find_packages
from setuptools import setup


setup(
    name='torram',
    version='0.0.1',
    packages=find_packages(),
    install_requires=('torch >= 1.11.0',
                      'torchvision >= 0.12.0',
                      'kornia == 0.6.2',
                      'randomname',
                      'matplotlib',
                      'pyyaml',
                      'pytest',
                      'tqdm'),
    author='Simon Schaefer',
    author_email='simon.k.schaefer@gmail.com',
    license='MIT',
    url='https://github.com/simon-schaefer/torram',
    description='Machine Learning essential tools for PyTorch.',
)
