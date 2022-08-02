from pathlib import Path
from setuptools import find_packages
from setuptools import setup

code_directory = Path(__file__).parent
long_description = (code_directory / "README.md").read_text()

setup(
    name='torram',
    version='0.0.4',
    packages=find_packages(),
    install_requires=('torch >= 1.11.0',
                      'numpy',
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
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/simon-schaefer/torram',
    description='Machine Learning essential tools for PyTorch.',
)
