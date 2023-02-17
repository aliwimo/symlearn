from setuptools import setup, find_packages
import symlearn

VERSION = symlearn.__version__
DESCRIPTION = symlearn.__doc__

setup(
    name="symlearn",
    version=VERSION,
    description=DESCRIPTION,
    long_description=DESCRIPTION,
    author="Mohamed Aliwi",
    author_email="aliwimo@gmail.com",
    license="MIT License",
    packages=find_packages(),
    install_requires=["numpy", "graphviz==0.16"],
    url="https://github.com/aliwimo/symlearn",
    keywords=["python", "symbolic", "regression", "optimization"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries"
    ]
)
