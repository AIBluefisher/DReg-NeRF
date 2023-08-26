#!/bin/sh
#SBATCH --job-name=installation
#SBATCH --output=/home/c/chenyu/log/%j.log
#SBATCH --error=/home/c/chenyu/log/%j.err

#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64000 # 64GB
#SBATCH --partition=medium
#SBATCH --nodelist=xgph5
#SBATCH --time=03:00:00 # days-hh;mm;ss
#SBATCH --cpus-per-task=8

echo "$state Start"
echo Time is `date`
echo "Directory is ${PWD}"
echo "This job runs on the following nodes: ${SLURM_JOB_NODELIST}"

# Install Pytorch
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch

# Install OpenBlas and then MinkowskiEngine
cd $HOME
mkdir openblas
# Without being a superuser, we have to install OpenBlas by compiling it from source.
cd $HOME/Projects
git clone https://github.com/xianyi/OpenBLAS.git
cd OpenBLAS
make
make install PREFIX=$HOME/openblas
# https://gist.github.com/bmmalone/1b5f9ff72754c7d4b313c0b044c42684
echo export LD_LIBRARY_PATH=$HOME/openblas:$LD_LIBRARY_PATH >> ~/.bashrc
echo export BLAS=$HOME/openblas/lib/libopenblas.a >> ~/.bashrc
echo export ATLAS=$HOME/openblas/lib/libopenblas.a >> ~/.bashrc
# Install MinkowskiEngine from source.
cd ../
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install --blas_include_dirs=$HOME/openblas/include --blas=openblas

# Install tiny-cuda-nn
pip uninstall tinycudann
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# Test if cuda available for torch and tcnn
python ./Collaborative-NeRF/scripts/test_cuda_available.py

# Open3D
pip install open3d

pip install tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg \
            kornia lpips tensorboard visdom tensorboardX matplotlib plyfile trimesh
