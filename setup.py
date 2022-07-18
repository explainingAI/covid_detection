from setuptools import setup, find_packages

description = """
Utilities to slice and split lung images with masks
"""

setup(
    name="uib_covid_detection",
    license="MIT",
    version="0.1",
    description=description,
    author="Francisco Arenas Afan de Rivera",
    author_email="franarenasafan@gmail.com",
    packages=find_packages(where='src'),
    python_requires=">=3.10",
    install_requires=[
        "uib-vfeatures>=0.7",
        "opencv-python>=4.5.5.64",
        "numpy>=1.22.4"
    ]
)
