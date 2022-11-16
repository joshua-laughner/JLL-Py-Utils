from setuptools import setup, find_packages

setup(
    name='JLLUtils',
    version='0.1',
    packages=find_packages(),
    install_requires=['cftime', 'matplotlib', 'netCDF4', 'numpy', 'pandas', 'pydap', 'scipy', 'statsmodels', 'xarray'],
    url='https://github.com/joshua-laughner/JLL-Py-Utils',
    license='',
    author='Joshua Laughner',
    author_email='jllacct119@gmail.com',
    description='Collection of general Python utilities'
)
