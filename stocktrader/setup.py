from setuptools import setup, find_packages

setup(
    name="StockTrader",
    version="0.1.0",
    description="Custom Gymnasium environments for Stock Trading",
    author="Davies Luo and Matthew Fisher",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        'gymnasium',
        'numpy',
    ],
    include_package_data=True
)
