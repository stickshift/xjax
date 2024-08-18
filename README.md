# Exploring Jax

# Getting Started

This setup works on an Ubuntu 20.04 machine running on a Cloud service like Google cloud or AWS. 

## Set up Ubuntu Environment 

```sh
# Install Python 3.8
sudo apt update
sudo apt install -y make python3 python3-pip

# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Set Python 3.8 as the system default
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1
sudo update-alternatives --config python3
sudo update-alternatives --config python

# Install Numpy
python3 -m pip install numpy
```

### Install NVIDIA libraries

### CUDA 12.0

```sh
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda
```

### cuTensor 2.0

```sh
wget https://developer.download.nvidia.com/compute/cutensor/2.0.0/local_installers/cutensor-local-repo-ubuntu2004-2.0.0_1.0-1_amd64.deb
sudo dpkg -i cutensor-local-repo-ubuntu2004-2.0.0_1.0-1_amd64.deb
sudo cp /var/cutensor-local-repo-ubuntu2004-2.0.0/cutensor-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install libcutensor2 libcutensor-dev libcutensor-doc
```

###  NCCL 2.17.1-CUDA 12.0

```sh
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt install libnccl2=2.17.1-1+cuda12.0 libnccl-dev=2.17.1-1+cuda12.0
```

## cuDNN 8.8-CUDA 12.0

```
https://developer.nvidia.com/downloads/c120-cudnn-local-repo-ubuntu2004-88012110-1amd64deb
```

## cuSPARSELt

```sh
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
sudo apt-get install libcusparselt0 libcusparselt-dev
```

Now switch Python from 3.8 to 3.12


```sh
# Install Python 3.12
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.12

# Set Python 3.12 as the system default
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
sudo update-alternatives --config python3

sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1
sudo update-alternatives --config python
```

# Install Python development libraries
```sh
sudo apt install python3.12-dev
```

### Apple Silicon

I had to pin libomp to 11.1.0 to avoid segfaults in pytorch.

```bash
curl -sL -o libomp.rb https://raw.githubusercontent.com/Homebrew/homebrew-core/fb8323f2b170bd4ae97e1bac9bf3e2983af3fdb0/Formula/libomp.rb
brew install ./libomp.rb
```

## Configure Environment

```bash
# Configure environment
source environment.sh
make

# Activate venv
source .venv/bin/activate

# Configure Jupyter 
jupyter lab --generate config

# Open the config file 
vim ~/.jupyter/jupyter_lab_config.py
```
Add or modify the following lines 

```py
c.ServerApp.ip = '0.0.0.0'
c.Serverapp.open_browser = False
c.ServerApp.port = 8888
```

```sh
# Launch jupyter
jupyter lab
```
