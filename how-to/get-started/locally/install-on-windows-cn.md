| title                             | summary                       | category | aliases                                               |
| --------------------------------- | ----------------------------- | -------- | ----------------------------------------------------- |
| 如何在 Windows 平台上安装 PyTorch | PyTorch 的 Windows 平台安装指南 | how-to   | /docs/how-to/get-started/locally/install-on-windows/ |




# 如何在 Windows 平台上安装 PyTorch

## 概述

本文档介绍如何在 Windows 平台上安装 PyTorch。

基本步骤如下：

1. 对应需求选择合适的 PyTorch 版本；
2. 确认机器是否已配置 CUDA;
3. 安装适用于 Windows 平台 的 PyTorch 分发包；
4. 测试以验证安装是否成功.



## 运行前提

> **注意：** PyTorch 只支持64位系统。

**操作系统要求**

| 操作系统 | 最低版本 | 建议版本 |
| :----------------------------------------------------------- | :------ | :---------- |
| [Windows](https://www.microsoft.com/en-us/windows)           | 7       | 10 或更高版本 |
| [Windows Server](https://docs.microsoft.com/en-us/windows-server/windows-server) | 2008 r2 |             |



**支持语言**

| 语言 | 支持版本  | 可用安装方式   |
| :------- | :------- | :------------------- |
| Python   | Python 3 | Conda / Pip / 源码 |
| C++      |          | LibTorch             |



## 第一步：选择 PyTorch 版本

请根据需求和条件选择合适的 PyTorch 版本。



#### 正式版 / 预览版

- **正式版**

当前正式发布的版本，适合多数用户。

> **注意：** 本文将以当前最新正式版 PyTorch 1.4.0 为例进行安装说明。

- **预览版**

下一个 PyTorch 重大更新版本的快速预览版，未经全面测试。



#### 仅需 CPU 版 / 支持 GPU 版 /

PyTorch 支持使用 GPU 加载模型运算，也能在仅使用CPU 的情况下运行。 

在现有版本中，仅需 CPU 版和支持 GPU 版是作为两种安装包分别发布的。


- **仅需 CPU 的 PyTorch 版本**

  此类版本不包含 CUDA 与 GPU 加速，省略了需要用到 GPU 的功能，体积较小。


- **支持 GPU 的 PyTorch 版本**

  此类版本能够启用[利用 GPU 加速](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html?highlight=cuda#cuda-tensors)的高级功能，体积较大。

  GPU 比 CPU 更适合深度学习大量运算的需求，一般建议安装支持 GPU 的 PyTorch 版本以获取更优的运算性能。PyTorch 只支持搭载 CUDA 技术的英伟达（NVIDIA）显卡启用 GPU 加速功能。
  




## 第二步: 配置 CUDA

> **注意：**  如果您的机器只安装了集成显卡或者显卡不支持 CUDA，请跳过此步骤。 



**硬件要求**

- [搭载 CUDA 技术的英伟达显卡](https://developer.nvidia.com/cuda-gpus )

  

**软件要求**

- [英伟达显卡驱动程序](https://www.nvidia.com/drivers) —— 安装 CUDA 10.1 需要 418.x 或更新版本的显卡驱动。

- [CUDA 工具包](https://developer.nvidia.com/cuda-toolkit-archive) —— PyTorch 1.4.0 支持的 CUDA 版本为 9.2 和 10.1。

  

**配置过程**

1. 确认上述硬件和软件要求；
2. [更新](https://www.nvidia.com/drivers)英伟达显卡驱动程序；
3. 若已安装 CUDA，确认CUDA 的版本信息符合要求。
   - 可使用 `nvcc --version` 命令行来获取CUDA 的版本信息。
4. 若尚未安装 CUDA，根据 [CUDA 安装手册](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows)进行下载安装。



## 第三步：安装 PyTorch

从预编译的二进制包和源代码中选择一种方式安装 PyTorch。

若无法明确需求，建议使用二进制包进行安装。

要使用二进制包安装，参见“使用二进制包”。

要从源代码构建 PyTorch，参见“从源代码编译”。

### 使用二进制包

通过包管理器从预构建的二进制包中安装 PyTorch 是体验最好的安装方法。


**准备环境**

建议配置 [Anaconda](https://docs.anaconda.com/anaconda/install/windows/) 环境。


#### 使用 Conda

**安装过程**

1. 打开 Anaconda 的命令行界面；

2. 根据下述情况分别执行相对应的命令行。

- 支持 GPU 的 PyTorch 版本

  若 CUDA 版本为 9.2：

  ```python
  conda install pytorch torchvision cudatoolkit=9.2 -c pytorch -c defaults -c numba/label/dev
  ```

  若 CUDA 版本为 10.1：

  ```python
  conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
  ```

- 仅需 CPU 的 PyTorch 版本 

  若配置不支持 CUDA：

  ```python
  conda install pytorch=0.4.1 -c pytorch
  ```



#### 使用 Pip

由于 [pip](https://pypi.org/project/pip) 的安装捆绑在 Python 的安装程序中，此处假定您的系统中已经可以使用 pip。

**安装过程**

1. 打开 Anaconda 的命令行界面或 Windows 系统的命令提示符；
2. 根据下述情况分别执行相对应的命令行。

- 支持 GPU 的 PyTorch 版本

  若 CUDA 版本为 9.2：

  ```python
  pip install torch==1.4.0+cu92 torchvision==0.5.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
  ```

  若 CUDA 版本为 10.1：

  ```python
  pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html
  ```

- 仅需 CPU 的 PyTorch 版本 

  若配置不支持 CUDA：

  ```python
  pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.htmlCPU-only 
  ```




#### 使用 LibTorch

*LibTorch* 是 PyTorch 的 C++ 分发包。 

**安装过程**

1. 打开 Anaconda 的命令行界面或 Windows 系统的命令提示符；
2. 根据下述情况执行相对应的命令行来下载和解压 LibTorch 包。

- 支持 GPU 的 PyTorch 版本

  若 CUDA 版本为 9.2：

  ```c++
  wget https://download.pytorch.org/libtorch/cu92/libtorch-win-shared-with-deps-1.4.0.zip
  unzip libtorch-win-shared-with-deps-latest.zip
  ```

  若 CUDA 版本为 10.1：

  ```c++
  wget https://download.pytorch.org/libtorch/cu101/libtorch-win-shared-with-deps-1.4.0.zip
  unzip libtorch-win-shared-with-deps-latest.zip
  ```

- 仅需 CPU 的 PyTorch 版本

  若配置不支持 CUDA：

  ```c++
  wget https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-latest.zip
  unzip libtorch-win-shared-with-deps-latest.zip
  ```



### 从源代码编译

在某些情况下，您可能需要从源代码编译  PyTorch 并将其安装在 Windows 上。

**准备环境**

确保您的机器上已安装：

- 支持 C++14 的编译器； 

- CMake 的生成器；

  - Visual Studio 2017 —— 要求高于 15.3.3 版本并配备 14.11 版本工具集；
  - Visual Studio 2019
  - [Ninja](https://ninja-build.org/)

- NVIDIA [Nsight Compute](https://developer.nvidia.com/nsight-compute-2019_5)；

  > **注意：** 在安装 CUDA 及 Nsight Compute 前，须完成 Visual Studio 2017 的安装。
  >
  > 由于 CUDA 与 MSVC 容易造成依赖版本冲突，请确保按照下表安装对应的 Visual Studio 工具链。

  | CUDA version | Newest supported VS version                             |
  | ------------ | ------------------------------------------------------- |
  | 9.0 / 9.1    | Visual Studio 2017 Update 4 (15.4) (`_MSC_VER` <= 1911) |
  | 9.2          | Visual Studio 2017 Update 5 (15.5) (`_MSC_VER` <= 1912) |
  | 10.0         | Visual Studio 2017 (15.X) (`_MSC_VER` < 1920)           |
  | 10.1         | Visual Studio 2019 (16.X) (`_MSC_VER` < 1930)           |

- [Anaconda](https://www.anaconda.com/distribution/#download-section) 环境。

详细说明可参考： https://github.com/pytorch/pytorch#from-source



**从源代码构建 PyTorch**

1. 输入以下命令行以安装依赖项：

   ```python
   conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing
   ```

   若 Python 版本低于 3.5, 需要先安装 `typing` 命令。

2. 运行以下命令以获取 PyTorch 源代码：

   ```python
   git clone --recursive https://github.com/pytorch/pytorch
   cd pytorch
   ```

   若是更新源码到本地，需要下载并更新第三方库。请执行：

   ```python
   git submodule sync
   git submodule update --init --recursive
   ```

3. 若系统已安装 Ninja，默认将使用 Ninja 作为 CMake 生成器。

   若使用其他生成器，使用下述命令行。

   - 若使用 Visual Studio 2017，请执行:

   ```python
   cmd
   set CMAKE_GENERATOR=Visual Studio 15 2017
   ```

   - 若使用 Visual Studio 2017，请执行:

   ```python
   cmd
   set CMAKE_GENERATOR=Visual Studio 16 2019
   ```

   若要在已安装 Ninja 的情况下指定其他生成器，首先执行 `set USE_NINJA=OFF` 命令。

4. （以下为可选操作，推荐执行；若使用 Python 3.5 则必须执行。）

   若要重写底层工具包，请添加以下命令行：

   > **注意：** 若使用 Visual Studio 进行该操作，请确保 CMake 的版本高于 3.12。
   
   ```python
   set CMAKE_GENERATOR_TOOLSET_VERSION=14.11
   set DISTUTILS_USE_SDK=1
   for /f "usebackq tokens=*" %i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -version [15^,16^) -products * -latest -property installationPath`) do call "%i\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=%CMAKE_GENERATOR_TOOLSET_VERSION%
   ```

5. （以下为可选操作，当版本号有较大差异时不推荐执行。）

   若要重写 CUDA 主机编译器，请将 `CUDAHOSTCXX`设置为：
   
   ```python
   set CUDAHOSTCXX=C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\VC\Tools\MSVC\14.11.25503\bin\HostX64\x64\cl.exe
   ```

6. 执行以下命令以安装 PyTorch：

   ```
   python setup.py install
   ```

   


## 第四步：验证安装

**样例代码测试**

为确保 PyTorch 安装成功，请按以下指导运行简单的样例代码测试。

- 输入以下命令行以进入 Python 的解释器:

  ```
  python
  ```

- 依次输入以下命令，打印出随机生成的张量矩阵:

  ```
  from __future__ import print_function
  import torch
  x = torch.rand(5, 3)
  print(x)
  ```

- 输出结果应类似于以下内容：

  ```
  tensor([[0.3380, 0.3845, 0.3217],
          [0.8337, 0.9050, 0.2650],
          [0.2979, 0.7141, 0.9069],
          [0.1449, 0.1132, 0.1375],
          [0.4675, 0.3947, 0.1426]])
  ```
  
  实际输出结果中张量的值无需与样例相同。
  

**GPU 可用性检查**

此外，请按照以下步骤检查 PyTorch 是否已启用并能访问 GPU 驱动程序和 CUDA。

- 执行以下命令:

  ```
  import torch
  torch.cuda.is_available()
  ```

- 输出结果应类似于以下内容：

  ```
  True
   0.6040  0.6647
   0.9286  0.4210
  [torch.FloatTensor of size 2x2]
  ```

  实际输出结果中张量的值无需与样例相同。

- 若 `cuda.is_available()` 返回 `False`, 请调试 CUDA 的安装，确保 PyTorch 检测到您的显卡。

  
