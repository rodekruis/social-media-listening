"""
Setup.py file.
Install once-off with:  "pip install ."
For development:        "pip install -e .[dev]"
"""
import setuptools


with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

PROJECT_NAME = "smm"

setuptools.setup(
    name=PROJECT_NAME,
    version="0.2",
    author="Phuoc Phùng, Wessel de Jong, Jacopo Margutti",
    author_email="pphung@redcross.nl, wdejong@redcross.nl, jmargutti@redcross.nl",
    description="Social media monitoring pipeline",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=install_requires,
    extras_require={
        "dev": [
            "black",
            "flake8"
        ],
    }
)