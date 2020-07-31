from setuptools import setup, find_packages

setup(
    name='nasim',
    version='0.5.0',
    url="https://networkattacksimulator.readthedocs.io",
    description="A simple and fast simulator for remote network pen-testing",
    long_description=open('README.rst').read(),
    long_description_content_type='text/x-rst',
    author="Jonathon Schwartz",
    author_email="Jonathon.Schwartz@anu.edu.au",
    license="MIT",
    packages=[
        package for package in find_packages()
        if package.startswith('nasim')
    ],
    install_requires=[
        'gym>=0.17.2',
        'numpy>=1.18.0',
        'networkx>=2.4',
        'matplotlib>=3.1.3',
        'pyyaml>=5.3',
        'prettytable>=0.7.2'
    ],
    extras_require={
        'dqn': [
            'torch>=1.5.0',
            'tensorboard>=2.2.2'
        ],
        'docs': [
            'sphinx>=3.0.4'
        ],
        'test': [
            'pytest>=5.4.3'
        ]
    },
    python_requires='>=3.7',
    package_data={
        'nasim': ['nasim/scenarios/benchmark/*.yaml']
    },
    project_urls={
        'Documentation': "https://networkattacksimulator.readthedocs.io",
        'Source': "https://github.com/Jjschwartz/NetworkAttackSimulator/",
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    zip_safe=False
)
