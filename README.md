# Pseudo-3D interpolation workflow

Interpolation workflow to generate pseudo-3D cubes from multiple 2D seismo-acoustic profiles.

This project is licensed under [`GNU GPLv3`](./LICENSE).

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

Install the workflow package in your active environment (with *optional* dependencies):

```bash
>>> pip install [-e] .           # -e: setuptools "develop mode"
>>> pip install ".[tide]"        # install optional "tpxo-tide-prediction" dependency
>>> pip install ".[transforms]"  # install optional transform dependencies
>>> pip install ".[extra]"       # install optional dependencies (geopandas, numba)
>>> pip install ".[tide,transforms,extra]"  # install all optional dependencies (RECOMMENDED!)
>>> pip install ".[dev]"         # developer installation
```

> **Note**
> 
> The required dependencies will be installed automatically if they are not already available.



> **Warning**
> 
> The **Curvelet transform** is only available on Unix systems via the [`curvelops`](https://github.com/PyLops/curvelops) package!  
> Please refer to the [README](https://github.com/PyLops/curvelops#readme) of the project and [additional installation instructions](https://github.com/DIG-Kaust/conda_envs/blob/main/install_curvelops.sh) if you plan to use this transform.

## Funding

This workflow was developed as part of my PhD research at the [School of Environment, University of Auckland, New Zealand](https://www.auckland.ac.nz/en/science/about-the-faculty/school-of-environment.html).

The funding was provided by the Royal Society of New Zealand Marsden Fund grant _Geologic champagne: What controls sudden release of CO2 at glacial terminations on the Chatham Rise?_ (19-UOA-339) 

Additional funding was issued by the _University of Auckland Doctoral Scholarship_.

## License

This project is licensed under [`GNU GPLv3`](./LICENSE). Please refer to the project [license](.LICENSE) when considering using this workflow for your own research or project.