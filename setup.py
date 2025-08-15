from setuptools import setup, find_packages

setup(
    name='pyescan',
    version='0.0.2',
    description='A package for dealing with retinal images exported by Private Eye',
    author='William Woof',
    author_email='awwoof@hotmail.com',
    packages=find_packages(),
    install_requires=[
        'click',
        'numpy',
        'pandas',
        'matplotlib',
        'tqdm'
    ],
    entry_points={
        'console_scripts': [
            'summarise_ce_export = pyescan.tools.cli:get_ce_export_summary_cli',
            'summarise_dataset = pyescan.tools.cli:summarise_dataset_cli',
            'run_function_on_csv = pyescan.tools.cli:run_function_on_csv_cli',
            'run_function_over_csv = pyescan.tools.cli:run_function_over_csv_cli',
            'run_metric = pyescan.tools.cli:run_metric_csv_cli',
            'narrow_to_wide = pyescan.tools.cli:narrow_to_wide_cli'
        ],
    },
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Libraries',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)