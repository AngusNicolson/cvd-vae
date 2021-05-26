
from setuptools import setup, find_packages

VERSION = "0.0.1"
DESCRIPTION = "VAE for CVD risk"

setup(
    name="cvd_vae",
    version=VERSION,
    author="Angus Nicolson",
    author_email="<angusjnicolson@gmail.com>",
    description=DESCRIPTION,
    packages=find_packages()
)
