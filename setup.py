from pathlib import Path

from setuptools import find_packages, setup

code_directory = Path(__file__).parent
long_description = (code_directory / "README.md").read_text()

setup(
    name="torram",
    packages=find_packages(),
    version="1.4.11",
    install_requires=(
        "torch >= 1.11.0",
        "numpy",
        "torchvision >= 0.12.0",
        "kornia",
        "omegaconf",
        "matplotlib",
        "pytest",
    ),
    author="Simon Schaefer",
    author_email="simon.k.schaefer@gmail.com",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/simon-schaefer/torram",
    description="Machine Learning essential tools for PyTorch.",
)
