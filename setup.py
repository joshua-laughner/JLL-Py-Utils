from setuptools import setup, find_packages

setup(
    name='JLLUtils',
    version='0.1',
    packages=find_packages(),
    install_requires=['matplotlib','numpy','pandas','scipy','statsmodels'],
    url='https://github.com/joshua-laughner/JLL-Py-Utils',
    license='',
    author='Joshua Laughner',
    author_email='jllacct119@gmail.com',
    description='Collection of general Python utilities'
)
