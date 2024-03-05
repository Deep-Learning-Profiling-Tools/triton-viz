from setuptools import setup, find_packages

setup(
    name="triton-viz",
    version="0.1",
    packages=find_packages(),
    description="A visualization tool for Triton",
    author="Deep Learning Profiling Tools Team",
    author_email="kzhou6@gmu.edu",
    url="https://github.com/Deep-Learning-Profiling-Tools/triton-viz",
    install_requires=[
        "setuptools",
        "triton",
        "gradio",
        "chalk-diagrams",
    ],
)
