""" Setup script for local developer install of `synd`. """

from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

setup(
    name='neurocode',
    version='0.1.0',
    author='Wilhelm Ågren',
    author_email='wilhelmagren98@gmail.com',
    packages=find_packages(),
    url='https://github.com/neurocode-ai/neurocode',
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'License :: OSI :: Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
    ],
    description='🧠 EEG/MEG self-supervised learning toolbox.',
    long_description=readme,
)
