from setuptools import setup

setup(
    name='MapDeduce',
    version='0.1.0',
    description='Handling antigenic maps and sequence data, testing amino '
                'acid polymorphisms associated with antigenicity.',
    author='David Pattinson',
    author_email='djp65@cam.ac.uk',
    packages=['MapDeduce'],
    install_requires=[
        "sklearn==0.0",
        "matplotlib==3.0.2",
        "spm1d==0.4.0",
        "tqdm==4.28.1",
        "biopython==1.72",
        "pandas==0.23.4",
        "glimix-core==1.3.7",
        "limix==1.0.12",
        "limix-core==1.0.2",
        "limix-legacy==0.8.12",
        "numpy>=1.14.5",
        "optimix==1.2.21",
        "rpy2==2.7.9",
        "seaborn==0.8.1"
    ]
)
