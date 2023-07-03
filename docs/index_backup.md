---
title: Introduction
date: 2023-05-31
hide:
  - footer
---

# Pseudo-3D cubes from densely spaced subbottom profiles via _Projection Onto Convex Sets_ (POCS) interpolation

This repository contains the source code for the generation of pseudo-3D cubes from densely spaced 2D subbottom profiles by utilizing the _Projection Onto Convex Sets_ ([POCS](./3D/3D_cube_interpolation/#pocs-theory)) method.

The source code accompanies a journal article published in ==[GEOPHYSICS]()==:

    XXXX et al. (2023) Pseudo-3D cubes from densely spaced subbottom profiles via Projection Onto Convex Sets interpolation: an open-source workflow applied to a pockmark field. GEOPHYSICS

!!! info "Usage information"

    When using this workflow please refer to the [citation information](citation.md) and, in case of feedback or questions, the [contact information](contact.md).

## Installation

### Preparations

The workflow can be installed locally after downloading and unzipping the source code from GitHub.

Navigate into the unzipped directory:

```bash
>>> cd ./pseudo-3D-interpolation  # root directory of unzipped package
```

### [Optional] Install dependencies using `conda`/`mamba`

Use the provided `conda`/`mamba` environment file to install dependencies:

```bash
>>> conda install -f {environment.yml}  # install dependencies
>>> conda activate pseudo_3d            # activate new env (default name: "pseudo_3d")
```

The repository includes **three** different environment YAML files:

- `environment.yml`: **Minimal dependencies** required to run the core workflow.

- `environment_optional.yml`: **Minimal and optional dependencies** required to run the full workflow:
  
     - [`tpxo-tide-prediction`](https://github.com/fwrnke/tpxo-tide-prediction) for **tide compensation**.
  
     - for **different** sparse **transforms**
       
          - **wavelet**: `pywavelets`
       
          - **shearlet**: `PyShearlets`
       
          - **curvelet**: see note below
  
     - `geopandas`: for QC
  
     - `numba`: for optimization

- `environment_dev.yml`: Developer dependencies (not for common user).

### [Optional] Install dependencies using `pip`

Use the provided `pip` requirements file to install dependencies:

```bash
>>> pip install -r {requirements.txt}   # install dependencies
```

The repository includes **three** different requirement files:

- `requirements.txt`: **Minimal dependencies** required to run the core workflow.

- `requirements_optional.txt`: **Minimal and optional dependencies** required to run the full workflow.
  
     - includes [`tpxo-tide-prediction`](https://github.com/fwrnke/tpxo-tide-prediction) for **tide compensation**.
  
     - for **different** sparse **transforms**
       
          - **wavelet**: `pywavelets`
       
          - **shearlet**: `PyShearlets`
       
          - **curvelet**: see note below
  
     - `geopandas`: for QC
  
     - `numba`: for optimization

- `requirements_dev.txt`: Developer dependencies (not for common user).

### Install from source

Install the workflow package in your active environment (with _optional_ dependencies):

```bash
>>> pip install [-e] .           # -e: setuptools "develop mode"
>>> pip install ".[tide]"        # install optional "tpxo-tide-prediction" dependency
>>> pip install ".[transforms]"  # install optional transform dependencies
>>> pip install ".[extra]"       # install optional dependencies (geopandas, numba)
>>> pip install ".[tide,transforms,extra]"  # install all optional dependencies (RECOMMENDED!)
>>> pip install ".[dev]"         # developer installation
```

!!! info "Dependencies"

    The required dependencies will be installed automatically if they are not already available.

!!! warning

    The **Curvelet transform** is only available on Unix systems via the [`curvelops`](https://github.com/PyLops/curvelops) package!  
    Please refer to the [README](https://github.com/PyLops/curvelops#readme) of the project and [additional installation instructions](https://github.com/DIG-Kaust/conda_envs/blob/main/install_curvelops.sh) if you plan to use this transform.

## License

This project is licensed under `GNU GPLv3`. Please refer to the project [license](license.md) when considering using this workflow for your own research or project.