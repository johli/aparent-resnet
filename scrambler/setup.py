import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="scrambler",
    version="0.2",
    author="Johannes Linder",
    author_email="johannes.linder@hotmail.com",
    description="Mask-based Interpretation for Variants",
    long_description=long_description,
    url="https://github.com/johli/aparent-resnet",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
