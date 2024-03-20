# Triton-Viz: A Visualization toolkit for GPU programming on Triton

<!-- PROJECT LOGO -->
<br />
<div align="center">
    <img src="Logo.jpg" alt="Logo" width="320" height="320">
</div>
<br/>
  
Welcome to Triton-Viz, a visualization and profiling toolkit designed for deep learning applications. Built with the intention of making GPU programming on Triton more intuitive.
<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about">About</a>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation-of-triton-viz">Installation of Triton_Viz</a></li>
      </ul>
    <li>
      <a href="#working with Examples">Working with Examples</a>
    <ul>
        <li><a href="#More puzzles">More Puzzles</a></li>
      </ul>
    </li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

## About

Triton-Viz is a visualization and analysis toolkit specifically designed to complement the development and optimization of applications written in OpenAI's Triton,an open-source programming language aimed at simplifying the task of coding for accelerators such as GPUs.
Triton-Viz emerges as a pivotal tool for developers working in the realms of AI and high-performance computing, offering a suite of features to enhance the debugging, performance analysis, and understanding of Triton code.
Given that Triton allows developers to program at a higher level while still targeting low-level accelerator devices, managing and optimizing resources like memory becomes a crucial aspect of development. 
Triton-Viz addresses these challenges by providing real-time visualization and profiling of tensor operations and their memory usage.Its interface is designed to be intuitive for users familiar with high-level array programming languages like Numpy and PyTorch.

The toolkit aids in identifying bottlenecks and inefficient memory operations,which are often the primary hurdles in achieving optimal performance on GPUs and other accelerators. 
By visualizing how Triton code translates into actual device-level operations, Triton-Viz empowers developers to make informed decisions about code structure, memory management, and parallel execution patterns. 
Whether you're a novice learning the intricacies of accelerator programming or an expert tuning algorithms for maximum efficiency,Triton-Viz serves as an essential tool for all. 

The best part about this tool is that while it does focus on visualizing GPU operations,users are not required to have GPU resources to run examples on their system.

## Getting Started

### Prerequisites
-Python installed(preferably the latest available version).
</br>
-[Triton](https://github.com/openai/triton/blob/main/README.md) installed.Follow the installation instructions in the linked repository.
</br>
</br>
-Upon successfully installing Triton,Install Torch using the following command
```sh
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
```
-Upon Succesful installation of Torch Make sure to uninstall 'pytorch-triton' using the following command.
```sh
pip uninstall pytorch-triton
```
### Installation of Triton_Viz
Clone the repository to your local machine:
```sh
git clone https://github.com/Deep-Learning-Profiling-Tools/triton-viz.git
cd triton-viz
pip install -e .
```
You're all set !

## Working with Examples:

```sh
cd examples 
python <file_name>.py
```
## More Puzzles
If you're interested in Fun Puzzles to work with in Triton, do check out:[Triton Puzzles](https://github.com/srush/Triton-Puzzles)

## License
Triton-Viz is open-sourced under the MIT License. See the See `LICENSE.txt` for details.
<p align="right">(<a href="#readme-top">back to top</a>)</p>

