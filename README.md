# tvt2022

## A quick guide to install the environment (Flocking - RL).

1. `apt update && apt install swig` (The version `SWIG Version 4.0.1` is ok.)
2. `cd env_wrapper`
3. `./reinstall`

The c++ implementation of the environment is placed in env_wrapper. 

The uav_2d_ma_fast.py is a python wrapper of the environment.

`td3_fast.py` is the first stage training script.

`sl_train.py` is the second stage training script.

Some results can be found in the `evaluation-fast.ipynb`.

Our models (parameters of the Neural Networks) can be downloaded from `https://box.nju.edu.cn/f/b668839eec3b478e9fd3/?dl=1`.
