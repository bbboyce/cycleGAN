from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pytorch-cyclegan-pix2pix",
    version="1.0.0",
    author="Jun-Yan Zhu, Taesung Park",
    description="PyTorch implementation of CycleGAN and pix2pix",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "torch>=2.4.0",
        "torchvision>=0.19.0",
        "numpy>=1.24.3",
        "scikit-image",
        "dominate>=2.8.0",
        "Pillow>=10.0.0",
        "wandb>=0.16.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)
