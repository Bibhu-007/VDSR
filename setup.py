from setuptools import setup, find_packages

# Read dependencies from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="image super-resolution",
    version="1.0.0",
    description="Single image super-resolution using VDSR.",
    author="Bibhu Prasad Bhanja",
    author_email="bibhu.p.bhanja@fau.de",
    python_requires=">=3.8",
    packages=find_packages(exclude=("tests", "docs")),
    install_requires=requirements,
    include_package_data=True,
    license="Apache-2.0",
)
