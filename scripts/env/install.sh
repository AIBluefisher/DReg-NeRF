# conda create -n dreg_nerf python=3.9
# conda activate dreg_nerf

# Torch 1.13 and CUDA 11.7
conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia

# Installing MinkowskiEngine by the script below:
# Ref: https://github.com/NVIDIA/MinkowskiEngine/issues/495
conda install openblas-devel -c anaconda
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine/
export CXX=g++-7
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
# For your convinience.
# Ref: https://github.com/NVIDIA/MinkowskiEngine/wiki/Installation
# > sudo apt install libopenblas-dev
# > pip install -U git+https://github.com/NVIDIA/MinkowskiEngine
#####################################################################################

# Tiny-cuda-cnn & nerfacc
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
# To install a specified version of nerfacc:
pip install nerfacc==0.3.5 -f https://nerfacc-bucket.s3.us-west-2.amazonaws.com/whl/torch-1.13.1_cu117.html

# Others.
pip install open3d tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg easydict \
            kornia lpips tensorboard visdom tensorboardX matplotlib plyfile trimesh h5py pandas \
            omegaconf PyMCubes Ninja
pip install -U scikit-learn
pip install git+https://github.com/jonbarron/robust_loss_pytorch
conda install pytorch-scatter -c pyg
