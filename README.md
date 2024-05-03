# mfs-chol

This repository contains the implementation for our SIGGRAPH 2024
paper
"[Lightning-fast Method of Fundamental Solutions](https://jiongchen.github.io/files/mfschol-paper.pdf)".

- **Build**: Our code was built on Ubuntu 20.04 with gcc 9.4 and CUDA
  12.1. To install other dependencies and compile the code, simply run
  ``build.sh``.

- **Get input data**: Download the input data from
  [here](https://drive.google.com/file/d/10pm4HebaCnEGdjCvEb2XhefkDxCgeFaB/view?usp=sharing). Extract
  the files and put them under ``dat/``.

- **Usage**: Bash scripts to replicate the results in our paper are
  provided. Run them from the main directory, for instance you may
  start with ``./run_some_extra.sh`` to solve for pixel
  diffusions. For GUI programs (e.g., Fig.10, 11, 13), press ``u`` to
  launch the solve.


## Contact 

If you find any compilation issues or bugs, please feel free to shoot
an email to jiong.chen@inria.fr.