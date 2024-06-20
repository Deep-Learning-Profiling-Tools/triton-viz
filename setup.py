from setuptools import setup, find_packages

setup(
    name="triton-viz",
    version="2.0",
    packages=find_packages(),
    include_package_data=True,
    description="A visualization tool for Triton",
    author="Deep Learning Profiling Tools Team",
    author_email="kzhou6@gmu.edu",
    url="https://github.com/Deep-Learning-Profiling-Tools/triton-viz",
    install_requires=[
        "setuptools",
        "triton",
        "gradio",
        "pyarrow",
        "pre-commit",
        "pytest",
    ],
)
