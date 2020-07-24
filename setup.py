"""Copyright (c) 2019 AIT Lab, ETH Zurich, Seonwook Park
""""""Setup module for GazeML."""

from setuptools import setup, find_packages

setup(
        name='gazeml',
        version='0.1',
        description='Data-driven gaze estimation using machine learning.',

        author='Seonwook Park',
        author_email='spark@inf.ethz.ch',

        packages=find_packages(exclude=[]),
        python_requires='>=3.5',
        install_requires=[
            'coloredlogs',
            'h5py',
            'numpy',
            'opencv-python',
            'pandas',
            'ujson',
            # Install the most appropriate version of Tensorflow
            # Ref. https://www.tensorflow.org/install/
            # 'tensorflow',
            # tensorflow-gpu==1.15
        ],
)
