from setuptools import setup, find_packages

setup(
    name="rdd_dl",
    version="1.0.0",
    description="Retinal Disease Detection Using Deep Learning and Explainable AI on OCT Images",
    author="Rajuzet",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "streamlit>=1.28.0",
        "tqdm>=4.65.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Programming Language :: Python :: 3.11",
    ],
)
