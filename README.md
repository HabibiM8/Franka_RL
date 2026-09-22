# Franka_RL (frankapp)

Reinforcement learning for the Franka Emika Panda on manipulation tasks (push, slide, pick & place).
Physics runs in [MuJoCo](https://mujoco.org/) on the CPU (one process per core), while the neural network
updates run on the GPU through [SBX](https://github.com/araffin/sbx) (Stable-Baselines3 in JAX).

The current baseline is **TQC + HER** on the goal-conditioned Franka environments from
[panda_mujoco_gym](https://github.com/zichunxx/panda_mujoco_gym).

---

## Requirements

- Linux (x86_64)
- NVIDIA GPU with a driver that supports **CUDA 12** (check with `nvidia-smi`)
- Python 3.12

You do **not** need a system-wide CUDA toolkit. `jax[cuda12]` installs the CUDA and cuDNN libraries as pip wheels
inside the virtual environment.

---

## Installation (GPU)

```bash

python3 -m venv env
source env/bin/activate

pip install --upgrade pip setuptools wheel
pip install -e .[dev,gpu]
```

```bash
python -c "
import gymnasium as gym, panda_mujoco_gym
env = gym.make('FrankaPickAndPlaceSparse-v0', render_mode='human')
env.reset()
for _ in range(200): env.step(env.action_space.sample())
env.close()
"
```

---

## Project structure

```
Franka_RL/
├── setup.cfg / setup.py     # package metadata and dependencies
├── frankapp/                # own code (wrappers, utilities, future tasks)
├── panda_mujoco_gym/        # vendored Franka MuJoCo environments (MIT, see below)
│   ├── __init__.py          # registers the env IDs with gymnasium
│   ├── envs/                # FrankaEnv base class + one subclass per task
│   └── assets/              # MuJoCo XML scenes (Menagerie Panda model)
├── experiments/
│   ├── train_tqc.py         # training: TQC + HER, 12 parallel envs
│   └── eval_tqc.py          # evaluation with viewer and success rate
├── tests/
└── logs/                    # checkpoints and eval logs (created on training)
```

---



---
---

## Citations

This project builds on the following work. Please cite them if you use this repository.

**Franka MuJoCo environments (panda_mujoco_gym)**

```bibtex
@misc{xu2023opensource,
  title         = {Open-Source Reinforcement Learning Environments Implemented in MuJoCo with Franka Manipulator},
  author        = {Zichun Xu and Yuntao Li and Xiaohang Yang and Zhiyuan Zhao and Lei Zhuang and Jingdong Zhao},
  year          = {2023},
  eprint        = {2312.13788},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO}
}
```

**MuJoCo**

```bibtex
@inproceedings{todorov2012mujoco,
  title     = {MuJoCo: A physics engine for model-based control},
  author    = {Todorov, Emanuel and Erez, Tom and Tassa, Yuval},
  booktitle = {2012 IEEE/RSJ International Conference on Intelligent Robots and Systems},
  pages     = {5026--5033},
  year      = {2012},
  doi       = {10.1109/IROS.2012.6386109}
}
```

**MuJoCo Menagerie (Franka Panda model)**

```bibtex
@software{menagerie2022github,
  title  = {{MuJoCo Menagerie: A collection of high-quality simulation models for MuJoCo}},
  author = {Zakka, Kevin and Tassa, Yuval and {MuJoCo Menagerie Contributors}},
  url    = {https://github.com/google-deepmind/mujoco_menagerie},
  year   = {2022}
}
```

**Gymnasium**

```bibtex
@article{towers2024gymnasium,
  title   = {Gymnasium: A Standard Interface for Reinforcement Learning Environments},
  author  = {Towers, Mark and Kwiatkowski, Ariel and Terry, Jordan and Balis, John U. and De Cola, Gianluca and Deleu, Tristan and Goul{\~a}o, Manuel and Kallinteris, Andreas and Krimmel, Markus and KG, Arjun and others},
  journal = {arXiv preprint arXiv:2407.17032},
  year    = {2024}
}
```

**Gymnasium-Robotics**

```bibtex
@software{gymnasium_robotics2023github,
  author  = {Rodrigo de Lazcano and Kallinteris Andreas and Jun Jet Tai and Seungjae Ryan Lee and Jordan Terry},
  title   = {Gymnasium Robotics},
  url     = {http://github.com/Farama-Foundation/Gymnasium-Robotics},
  version = {1.3.1},
  year    = {2024}
}
```

**Stable-Baselines3**

```bibtex
@article{raffin2021sb3,
  title   = {Stable-Baselines3: Reliable Reinforcement Learning Implementations},
  author  = {Antonin Raffin and Ashley Hill and Adam Gleave and Anssi Kanervisto and Maximilian Ernestus and Noah Dormann},
  journal = {Journal of Machine Learning Research},
  year    = {2021},
  volume  = {22},
  number  = {268},
  pages   = {1--8},
  url     = {http://jmlr.org/papers/v22/20-1364.html}
}
```

**SBX (Stable-Baselines3 in JAX)**: https://github.com/araffin/sbx, by Antonin Raffin (cite together with SB3).

**JAX**

```bibtex
@software{jax2018github,
  author  = {James Bradbury and Roy Frostig and Peter Hawkins and Matthew James Johnson and Chris Leary and Dougal Maclaurin and George Necula and Adam Paszke and Jake Vander{P}las and Skye Wanderman-{M}ilne and Qiao Zhang},
  title   = {{JAX}: composable transformations of {P}ython+{N}um{P}y programs},
  url     = {http://github.com/jax-ml/jax},
  year    = {2018}
}
```

**Flax**

```bibtex
@software{flax2020github,
  author  = {Jonathan Heek and Anselm Levskaya and Avital Oliver and Marvin Ritter and Bertrand Rondepierre and Andreas Steiner and Marc van {Z}ee},
  title   = {{F}lax: A neural network library and ecosystem for {JAX}},
  url     = {http://github.com/google/flax},
  year    = {2024}
}
```

**TQC (algorithm)**

```bibtex
@inproceedings{kuznetsov2020tqc,
  title     = {Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics},
  author    = {Kuznetsov, Arsenii and Shvechikov, Pavel and Grishin, Alexander and Vetrov, Dmitry},
  booktitle = {Proceedings of the 37th International Conference on Machine Learning (ICML)},
  pages     = {5556--5566},
  year      = {2020}
}
```

**Hindsight Experience Replay**

```bibtex
@inproceedings{andrychowicz2017her,
  title     = {Hindsight Experience Replay},
  author    = {Andrychowicz, Marcin and Wolski, Filip and Ray, Alex and Schneider, Jonas and Fong, Rachel and Welinder, Peter and McGrew, Bob and Tobin, Josh and Abbeel, Pieter and Zaremba, Wojciech},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {30},
  year      = {2017}
}
```

---

## License

MIT. The vendored `panda_mujoco_gym` is also MIT-licensed; its original license file is kept in `panda_mujoco_gym/LICENSE`.
