<a name="readme-top"></a>
# Triton-Viz: A Visualization Toolkit for programming with Triton
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
      <a href="#About">About</a>
    <li>
      <a href="#Getting-Started">Getting Started</a>
      <ul>
        <li><a href="#Prerequisites">Prerequisites</a></li>
        <li><a href="#Installation-of-Triton_Viz">Installation of Triton_Viz</a></li>
      </ul>
    <li>
      <a href="#Working-with-Examples">Working with examples</a>
    <ul>
        <li><a href="#More-Puzzles">More puzzles</a></li>
      </ul>
    </li>
    <li><a href="#License">License</a></li>
  </ol>
</details>

## About

Triton-Viz is a visualization and analysis toolkit specifically designed to complement the development and optimization of applications written in OpenAI's Triton, an open-source programming language aimed at simplifying the task of coding for accelerators such as GPUs.
Triton-Viz offers a suite of features to enhance the debugging, performance analysis, and understanding of Triton code.

Given that Triton allows developers to program at a higher level while still targeting low-level accelerator devices, managing and optimizing resources like memory becomes a crucial aspect of development.
Triton-Viz addresses these challenges by providing real-time visualization of tensor operations and their memory usage.
The best part about this tool is that while it does focus on visualizing GPU operations, users are not required to have GPU resources to run examples on their system.

## Getting Started

### Prerequisites

- Python installed (preferably the latest available version), minimum supported version is 3.10.
- [Triton](https://github.com/openai/triton/blob/main/README.md) installed. Follow the installation instructions in the linked repository.
- Note: the below commands must be run in order.

Triton install (need nightly):
```
pip install -U triton --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/
```

Upon successfully installing Triton, install Torch using the following command:

```sh
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
```

Upon successful installation of Torch make sure to uninstall `pytorch-triton` using the following command:

```sh
pip uninstall pytorch-triton
```

### Installation of Triton-Viz

Clone the repository to your local machine:

```sh
git clone https://github.com/Deep-Learning-Profiling-Tools/triton-viz.git
cd triton-viz
pip install -e .
```

You're all set!

### Optional: Enable NKI Support

If you want to exercise the Neuron Kernel Interface (NKI) interpreter or run the NKI-specific tests:

1. Follow the [AWS Neuron Torch-NeuronX Ubuntu 22.04 setup guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/setup/neuron-setup/pytorch/neuronx/ubuntu/torch-neuronx-ubuntu22.html#setup-torch-neuronx-ubuntu22) to add the Neuron APT repository and install the required system packages (for example `aws-neuronx-tools`, `aws-neuronx-runtime-lib`, `aws-neuronx-collectives`, and their dependencies).
2. Instead of running `pip install -e .` in the above section, install Triton-Viz with the optional NKI extras so the Neuron Python packages (`neuronx-cc`, `libneuronxla`, `torch-neuronx`) are available:

   ```sh
   pip install -e .[nki]
   # or pip install triton-viz[nki]
   ```

### Testing
* To run core Triton-viz tests, run `pytest tests/`.
* (if NKI installed) To run NKI-specific tests, run `pytest tests/ -m nki`.
* To run all tests (Triton + NKI), run `pytest tests/ -m ""`.

## Working with Examples

```sh
cd examples
python <file_name>.py
```

### CPU interpreter concurrency

When running with the Triton CPU interpreter (`TRITON_INTERPRET=1`), you can emulate concurrent SMs by setting how many blocks execute in parallel:

```sh
export TRITON_VIZ_NUM_SMS=4  # or set triton_viz.config.num_sms in Python
```

This is useful for kernels that rely on cross-block synchronization (e.g., producer/consumer patterns) when testing without a GPU.

## More Puzzles

If you're interested in fun puzzles to work with in Triton, do check out: [Triton Puzzles](https://github.com/srush/Triton-Puzzles)

## License

Triton-Viz is licensed under the MIT License. See the [LICENSE](LICENSE) for details.

## Publication
If you find this repo useful for your research, please cite our paper:

```
@inproceedings{ramesh2025tritonviz,
  author={Ramesh, Tejas and Rush, Alexander and Liu, Xu and Yin, Binqian and Zhou, Keren and Jiao, Shuyin},
  title={Triton-Viz: Visualizing GPU Programming in AI Courses},
  booktitle = {Proceedings of the 56th ACM Technical Symposium on Computer Science Education (SIGCSE TS '25)},
  numpages = {7},
  location = {Pittsburgh, Pennsylvania, United States},
  series = {SIGCSE TS '25}
}
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>
