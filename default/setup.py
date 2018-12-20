from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['tensorflow==1.12.0', 'pandas==0.23.4','numpy==1.15.4','scipy==1.2.0','google-cloud-storage','sh']

setup(
    name='default',
    version='1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='The basic ML model to predict items based on past purchases.'
)
