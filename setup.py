from setuptools import find_packages, setup

setup(
    name="credit_screening",
    version="0.1.0",
    description="Japanese Credit Screening Risk Assessment System",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "joblib>=1.0.0",
        "statsmodels>=0.13.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.5b2",
            "flake8>=3.9.2",
            "isort>=5.9.1",
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
        ],
        "deploy": [
            "flask>=2.0.1",
            "gunicorn>=20.1.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="credit, risk assessment, machine learning, finance",
    project_urls={
        "Source": "https://github.com/yourusername/japanese-credit-screening",
        "Bug Reports": "https://github.com/yourusername/japanese-credit-screening/issues",
    },
)
