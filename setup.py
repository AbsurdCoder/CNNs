"""
Setup script for CNN Models Library
Install with: pip install -e .
"""

from setuptools import setup, find_packages
import os

# Read long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="cnn-models-library",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Comprehensive implementation of 10 landmark CNN architectures with unified comparison API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/cnn-models-library",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.2.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=6.1.0",
            "sphinx-rtd-theme>=1.2.0",
            "sphinx-autodoc-typehints>=1.22.0",
        ],
        "serving": [
            "fastapi>=0.95.0",
            "uvicorn>=0.21.0",
            "python-multipart>=0.0.6",
        ],
        "viz": [
            "plotly>=5.14.0",
            "dash>=2.9.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "cnn-compare=cnn_comparison_api:main",
            "cnn-train=train_pipeline:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.md"],
    },
    keywords="cnn deep-learning computer-vision pytorch neural-networks",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/cnn-models-library/issues",
        "Documentation": "https://cnn-models-library.readthedocs.io/",
        "Source": "https://github.com/yourusername/cnn-models-library",
    },
)