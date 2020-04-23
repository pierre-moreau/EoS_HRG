#!/usr/bin/env python

from setuptools import setup

def version():
    with open('EoS_HRG/__init__.py', 'r') as f:
        for line in f:
            if line.startswith('__version__ = '):
                return line.split("'")[1]
    raise RuntimeError('unable to determine version')

def requirements():
    req = []
    with open('requirements.txt', 'r') as f:
        for line in f:
            str_line = line.rstrip('\n')
            req.append(str_line)
    return req
    
def long_description():
    with open('README.md') as f:
        return f.read()

setup(
    name='EoS_HRG',
    version=version(),
    description='Equation of state (EoS) from a matching between lattice QCD (lQCD) and the Hadron Resonance Gas model (HRG)',
    long_description=long_description(),
    author='Pierre Moreau',
    author_email='pierre.moreau@duke.edu',
    url='https://github.com/pierre-moreau/EoS_HRG',
    license='MIT',
    packages=['EoS_HRG','EoS_HRG.test'],
    install_requires=requirements(),
    package_data={'EoS_HRG':['data/*.csv', '*.dat']},
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Physics'
    ]
)