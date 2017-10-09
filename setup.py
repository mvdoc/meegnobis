from distutils.core import setup

REQUIRED_PACKAGES = [
    'scipy',
    'numpy',
    'scikit-learn',
    'mne',
    'joblib'
]

EXTRA_PACKAGES = {
    'test': ['pytest']
}

setup(
    name='MEEGnobis',
    version='0.0.1dev',
    packages=['meegnobis'],
    license='Apache License 2.0',
    long_description=open('README').read(),
    author='Matteo Visconti di Oleggio Castello',
    author_email='mvdoc.gr@dartmouth.edu',
    description='Representational Similarity Analysis using noise-normalized '
                'cross-validated metrics for M/EEG',
    install_requires=REQUIRED_PACKAGES,
    extras_require=EXTRA_PACKAGES
)
