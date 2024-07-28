from setuptools import setup, find_packages

setup(
    name="stocktrader",
    version="0.1.0",
    description="Custom Gymnasium environments for Stock Trading",
    author="Davies Luo and Matthew Fisher",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        'gymnasium',
        'numpy',
        'torch',
    ],
    include_package_data=True
)
