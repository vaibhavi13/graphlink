from setuptools import find_packages, setup

setup(
    name='gmaze',
    packages=find_packages(include=['gmaze']),
    version='0.1.0',
    package_data={'gmaze': ['kernel.cu']},
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    description='Library for parallel clustering algorithms',
    author='Mana Agrawal',
    license='None',
    test_suite='tests',
)
# 'numba','numpy','cupy'