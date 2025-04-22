import os
from setuptools import setup, find_packages

# Read the contents of your README file
def read(fname):
    try:
        return open(os.path.join(os.path.dirname(__file__), fname)).read()
    except FileNotFoundError:
        return "Extreme Adverse Visual Vision - Object detection in adverse weather conditions"

# Define required packages
requires = [
    # Web application dependencies
    "flask",
    "flask-cors",
    "gunicorn",  # For production deployment
    "werkzeug",
    
    # Core ML dependencies
    "torch>=2.0.0",
    "ultralytics>=8.0.0",
    "numpy>=1.20.0",
    "opencv-python-headless>=4.5.0",
    "pillow>=8.0.0",
    
    # Data processing and visualization
    "matplotlib>=3.5.0",
    "pandas>=1.3.0",
    "tqdm>=4.60.0",
    "seaborn>=0.11.0",
    "pyyaml>=6.0",
]

setup(
    name="eavv",  # Name of your package
    version="0.1.0",  # Initial version
    author="Harshitha",  # Your name
    author_email="harshir07@gmail.com",  # Your email
    description="Extreme Adverse Visual Vision - Object detection in adverse weather conditions",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/harshitha-ras/EAVV",  # URL to your repository
    packages=find_packages(include=["eavv", "eavv.*"]),
    include_package_data=True,
    install_requires=requires,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Web Environment",
        "Framework :: Flask",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.5b2",
            "flake8>=3.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "eavv-train=eavv.train_yolov8:main",
            "eavv-convert=eavv.convert_xml_to_yolo:main",
            "eavv-analyze=eavv.apply_bodem:main",
            "eavv-web=eavv.app:main",
        ],
    },
)
