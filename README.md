# sdf

## Prerequisites
Python env:
- torch 2.1
- pymeshlab-2022.2
- bpy 3.4.0

Other combinations might also work

In case you don't have `libxkbcommon` installed:
- `conda install conda-forge::libxkbcommon` or use package manager
- in case of condas: `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib`

## Running train.sh on SGE

qsub train_siren.sh
