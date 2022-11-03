# tvt2022

- A quick guide to install the environment (Flocking - RL).

1. `apt update && apt install swig` (The version `SWIG Version 4.0.1` is ok.)
2. `cd env_wrapper`
3. `./reinstall`

---

- A quick guide to train td3 agent.

Some python packages may be needed, pls install these packages by `pip install`. (e.g. `pip install torch`), and then

1. `python td3_fast.py --agents=16 --folder=./logs --global_="GLOBAL"`

---

If you find this work is helpful at some aspects, please considering cite our paper:

W. Wang, L. Wang, J. Wu, X. Tao and H. Wu, "Oracle-Guided Deep Reinforcement Learning for Large-Scale Multi-UAVs Flocking and Navigation," in IEEE Transactions on Vehicular Technology, vol. 71, no. 10, pp. 10280-10292, Oct. 2022, doi: 10.1109/TVT.2022.3184043.

---
The c++ implementation of the environment is placed in env_wrapper. 

The uav_2d_ma_fast.py is a python wrapper of the environment.

`td3_fast.py` is the first stage training script.

`sl_train.py` is the second stage training script.

Some results can be found in the `evaluation-fast.ipynb`.

Our models (parameters of the Neural Networks) can be downloaded from `https://box.nju.edu.cn/f/b668839eec3b478e9fd3/?dl=1`.
