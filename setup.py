import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="snapy",
    use_scm_version=True,
    author="Daniel Precioso",
    author_email="daniel.precioso@uca.es",
    description="SNAPy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/daniprec/SNAPy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5, <3.7',
    setup_requires=['setuptools_scm'],
    install_requires=[
        'numpy<1.19.0',
        'typer==0.3.2',
        'matplotlib==3.3.2',
        'scipy==1.2.0'
    ]
)
