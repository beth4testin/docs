| title                          | summary                     | category | aliases                                            |
| ------------------------------ | --------------------------- | -------- | -------------------------------------------------- |
| Installation Guide for Windows | Install PyTorch on Windows. | how-to   | /docs/how-to/get-started/locally/install-on-windows |




# **Installation Guide for Windows**

## Overview

This guide explains how to install PyTorch under Microsoft Windows. 

The setup of PyTorch for Windows consists of a few simple steps:

1. Choose a suitable version of PyTorch to download.
2. Verify that the system is CUDA-capable.
3. Install the Windows-specific distribution of PyTorch. 
4. Test the Installation.



## Prerequisites

**Applicable for:** PyTorch 1.4.0 (Stable)

> **Note:** PyTorch is supported only on 64-bit system. 

**Operating system requirements**

| Operating System                                             | Minimum | Recommended |
| :----------------------------------------------------------- | :------ | :---------- |
| [Windows](https://www.microsoft.com/en-us/windows)           | 7       | 10 or above |
| [Windows Server](https://docs.microsoft.com/en-us/windows-server/windows-server) | 2008 r2 |             |



**Supported language**

| Language | Version  | Packages Available   |
| :------- | :------- | :------------------- |
| Python   | Python 3 | Conda / Pip / Source |
| C++      |          | LibTorch             |



## Step 1: Choose the proper version

Choose the proper version and variant of PyTorch to download.



#### Stable / Preview version

First, decide whether to get a stable release or a pre-release build.

- **Stable **

The most currently tested and supported version of PyTorch, suitable for many users. 

> **Note:** The following sections describe how to install  PyTorch 1.4.0, the current stable release.

- **Preview **

The nightly builds of the next major release of PyTorch, not fully tested and supported. 



#### CPU-only / GPU-enabled variant

PyTorch runs both in CPU-only or GPU-powered mode. 

For current releases,  CPU-only and GPU-enabled packages are separate.

- **CPU-only **

  The CPU-only variant is built without CUDA and GPU support. It has a smaller installation size, and omits features that would require a GPU. 

  > **Note:** GPU support is only available with NVIDIA CUDA-enabled cards on Windows.

- **GPU-enabled**

  The GPU-enabled variant pulls in CUDA and other NVIDIA components during install. It has larger installation size and includes [support](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html?highlight=cuda#cuda-tensors) for advanced features that require GPU.

  GPUs shine compared to CPUs for larger batch sizes.
  
  
  
  



## Step 2: Set up with GPU support

> **Note:** If you do not require a GPU-enabled version of PyTorch, skip this step.



**Hardware requirements**

- NVIDIA GPU card with CUDA Compute Capability. 

  - See the list of [CUDA-enabled GPU cards](https://developer.nvidia.com/cuda-gpus).
  
  

**Software requirements**

- [NVIDIA GPU drivers](https://www.nvidia.com/drivers) —CUDA 10.1 requires 418.x or higher.

- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) —PyTorch 1.4.0 supports CUDA 9.2 and 10.1.

  

**Setting up with CUDA**

1. Check the requirements listed above. 
2. [Get](https://www.nvidia.com/drivers) the latest NVIDIA driver.
3. If CUDA is installed, make sure it matches the versions listed above. 
   - To check for the CUDA version, use `nvcc --version`.
4. If not, read the [CUDA install guide for Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/) and install CUDA.



## Step 3: Install PyTorch

Choose from pre-packaged distributions containing binary (precompiled) programs or source code to install PyTorch. 

When in doubt, use a binary distribution.

For binary distributions, see the instructions in “From Binaries”.

To build PyTorch from source, use the instructions in “From Source”.



### From binary

For most use cases, installing from a pre-built binary via a package manager provides the best experience. 



**Prepare**

We highly recommend installing an [Anaconda](https://docs.anaconda.com/anaconda/install/windows/) environment first. 

Anaconda provides all of the PyTorch dependencies in one, sandboxed install, including Python and `pip.` 



#### Via conda

**Installing**

1. Open a Anaconda Prompt.

2. Use the commands as described below.

- GPU-enabled

  For systems with CUDA 9.2, run:

  ```python
  conda install pytorch torchvision cudatoolkit=9.2 -c pytorch -c defaults -c numba/label/dev
  ```

  For systems with CUDA 10.1, run:

  ```python
  conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
  ```

- CPU-only 

  For systems without CUDA and GPU support, run:

  ```python
  conda install pytorch=0.4.1 -c pytorch
  ```



#### Via pip

Since [pip](https://pypi.org/project/pip) comes bundled with Python installer, we assume that you already have it installed in your system.

**Installing**

1. Open a Anaconda Prompt or Windows Command Prompt.
2. Use the commands as described below.

- GPU-enabled

  For systems with CUDA 9.2, run:

  ```python
  pip install torch==1.4.0+cu92 torchvision==0.5.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
  ```

  For systems with CUDA 10.1, run:

  ```python
  pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html
  ```

- CPU-only 

  For systems without CUDA and GPU support, run:

  ```python
  pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.htmlCPU-only 
  ```




#### Via LibTorch

*LibTorch* is the C++ distribution of PyTorch.

**Installing**

1. Open a Anaconda Prompt or Windows Command Prompt.
2. Use the commands as described below.

- GPU-enabled

  For systems with CUDA 9.2, run:

  ```c++
  wget https://download.pytorch.org/libtorch/cu92/libtorch-win-shared-with-deps-1.4.0.zip
  unzip libtorch-win-shared-with-deps-latest.zip
  ```

  For systems with CUDA 10.1, run:

  ```c++
  wget https://download.pytorch.org/libtorch/cu101/libtorch-win-shared-with-deps-1.4.0.zip
  unzip libtorch-win-shared-with-deps-latest.zip
  ```

- CPU-only

  For systems without CUDA and GPU support, run:

  ```c++
  wget https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-latest.zip
  unzip libtorch-win-shared-with-deps-latest.zip
  ```



### From source

Under some circumstances, it may be preferable to install from a source distribution.

**Prepare**

To build PyTorch from source, you will need:

- A C++14 compiler. 

- A generator of CMake:

  - Visual Studio 2017 ( version 15.3.3 or above, with the toolset 14.11) 
  - Visual Studio 2019
  - [Ninja](https://ninja-build.org/)

- NVIDIA [Nsight Compute](https://developer.nvidia.com/nsight-compute-2019_5).

  > **Note:** Be sure that CUDA with Nsight Compute is installed after Visual Studio 2017.
  
- CUDA and MSVC have strong version dependencies.

  Install the corresponding VS toolchain in the table below.

  | CUDA version | Newest supported VS version                             |
  | ------------ | ------------------------------------------------------- |
  | 9.0 / 9.1    | Visual Studio 2017 Update 4 (15.4) (`_MSC_VER` <= 1911) |
  | 9.2          | Visual Studio 2017 Update 5 (15.5) (`_MSC_VER` <= 1912) |
  | 10.0         | Visual Studio 2017 (15.X) (`_MSC_VER` < 1920)           |
  | 10.1         | Visual Studio 2019 (16.X) (`_MSC_VER` < 1930)           |

- Also, we highly recommend installing an [Anaconda](https://www.anaconda.com/distribution/#download-section) environment.

See also: https://github.com/pytorch/pytorch#from-source



**Build PyTorch from source**

1. To install Dependencies, run:

   ```python
   conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing
   ```

   For Python version lower than 3.5, install `typing` first.

2. To get the PyTorch Source, run:

   ```python
   git clone --recursive https://github.com/pytorch/pytorch
   cd pytorch
   ```

   If you are updating an existing checkout:

   ```python
   git submodule sync
   git submodule update --init --recursive
   ```

3. Ninja will be the default generator if installed.

   Otherwise, add the lines as described below.

   - To build with VS 2017 generator:

   ```python
   cmd
   set CMAKE_GENERATOR=Visual Studio 15 2017
   ```

   - To build with VS 2019 generator:

   ```python
   cmd
   set CMAKE_GENERATOR=Visual Studio 16 2019
   ```

   To specify other generator when Ninja is installed, use `set USE_NINJA=OFF`.

4.  *(Optional but recommended for most cases; essential for Python 3.5 users)* To override the underlying toolset, add these lines:

   > **Note:** For a Visual Studio generator to do this, the minimum required version of CMake is 3.12.
   
   ```python
set CMAKE_GENERATOR_TOOLSET_VERSION=14.11
   set DISTUTILS_USE_SDK=1
   for /f "usebackq tokens=*" %i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -version [15^,16^) -products * -latest -property installationPath`) do call "%i\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=%CMAKE_GENERATOR_TOOLSET_VERSION%
   ```
   
5.  *(Optional, but not recommended if there are big version differences)* To override the CUDA host compiler, set `CUDAHOSTCXX` :

   ```python
   set CUDAHOSTCXX=C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\VC\Tools\MSVC\14.11.25503\bin\HostX64\x64\cl.exe
   ```

6. To install PyTorch, run:

   ```
   python setup.py install
   ```

   


## Step 4: Verify the installation

**Verifying the installation**

To ensure that PyTorch was installed correctly, run the sample PyTorch code as follows. Here we will construct a randomly initialized tensor.

- From the command line, type:

  ```
  python
  ```

- Then enter the following code:

  ```
  from __future__ import print_function
  import torch
  x = torch.rand(5, 3)
  print(x)
  ```

- The output should be something similar to this:

  ```
  tensor([[0.3380, 0.3845, 0.3217],
          [0.8337, 0.9050, 0.2650],
          [0.2979, 0.7141, 0.9069],
          [0.1449, 0.1132, 0.1375],
          [0.4675, 0.3947, 0.1426]])
  ```

  

**Verifying the CUDA driver**

Additionally, check if your GPU driver and CUDA is enabled and accessible by PyTorch as follows.

- From the command line, type:

  ```
  import torch
  torch.cuda.is_available()
  ```

- This should produce output similar to this:

  ```
  True
   0.6040  0.6647
   0.9286  0.4210
  [torch.FloatTensor of size 2x2]
  ```

  The values of the tensor will be different on your instance. 

- If `cuda.is_available()` returns `False`, debug your CUDA installation so PyTorch can see your graphics card. 

  
