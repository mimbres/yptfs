from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='mockmt3',
    version='0.1',
    packages=find_packages(),
    install_requires=[req for req in requirements if 'git+' not in req],
    dependency_links=[req for req in requirements if 'git+' in req],
    author='https://github.com/mimbres',
    description='YourMT3: Multi-task and multi-track music transcription for everyone',
    license='Apache 2.0',
)
