from setuptools import setup, find_packages

setup(
    name="enhanced-autonomous-vehicle-vision",
    version="0.1.0",
    packages=find_packages(),
    
    # Dependencies required for visualization and analysis
    install_requires=[
        'matplotlib>=3.0.0',  # For creating plots
        'pandas>=1.0.0',      # For data manipulation
        'numpy>=1.18.0',      # For numerical operations
        'seaborn>=0.11.0'     # For hexbin plots and advanced visualizations
    ],
    
    # Metadata
    author="Harshitha Rasamsetty",
    author_email="harshir07@gmail.com",
    description="Enhanced Autonomous Vehicle Vision (EAVV): A merged dataset combining DAWN and WEDGE for adverse weather conditions",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    keywords=[
        "computer-vision",
        "autonomous-vehicles",
        "weather-conditions",
        "object-detection",
        "DAWN",
        "WEDGE",
        "adverse-weather",
        "synthetic-data"
    ],
    url="https://github.com/harshitha-ras/EAVV",
    
    # Classifiers
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    
    # Package Structure
    package_data={
        'eavv': [
            'EAVV.zip',    # Combined dataset
            'format.py',   # XML standardization script
            'EDA.py'       # Exploratory Data Analysis script
        ],
    },
    
    # Entry points for command-line scripts
    entry_points={
        'console_scripts': [
            'eavv-format=eavv.format:main',
            'eavv-analyze=eavv.EDA:main',
        ],
    },
    
    # Python version requirement
    python_requires='>=3.7',
)