from setuptools import setup, find_packages

setup(
    name="pollution_model_3d",
    version="0.1.8",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
    ],
    python_requires=">=3.8",
) 