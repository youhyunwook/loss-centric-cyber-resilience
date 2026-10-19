"""
setup.py
--------
Editable install: ``pip install -e .``
"""
from setuptools import setup, find_packages

setup(
    name="cyber_resilience_framework",
    version="1.0.3",
    description=(
        "Loss-Centric Cyber Resilience Evaluation and "
        "Automated Countermeasure Ranking Using Public Network Flow Data — IEEE Access reproducibility package."
    ),
    author="Hyunwook You, Jisoo Jang, Minjin Choi, Jina Lee, Haneul Ryu, Dongkyoo Shin",
    packages=find_packages(exclude=("tests", "experiments", "scripts")),
    python_requires=">=3.10",
    install_requires=[
        "pandas>=2.2.3",
        "numpy>=2.1.3",
        "scipy>=1.10.0",
        "networkx>=3.4.2",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.3.0",
        "click>=8.1.0",
        "rich>=13.0.0",
    ],
)
