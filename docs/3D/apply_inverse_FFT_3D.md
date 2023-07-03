---
title: "Step 14: Frequency to time domain conversion [3D]"
date: 2022-08-12
hide:
  - footer
---

# Frequency to time domain conversion
Apply a _frequency_ to _time_ domain conversion using a 1D Fast Fourier Transform (FFT) along the time axis of an input 3D cube.

## Description
This script uses [`xrft.xrft.ifft`](https://xrft.readthedocs.io/en/latest/api.html#xrft.xrft.ifft) to compute the inverse FFT. `xrft` is build on `xarray`, `dask` and `numpy` and preserves netCDF metadata during computations.

### Command line interface
The script needs a **single netCDF** (3D) as input (in _frequency_ domain):

There are two options to run the script. We recommend using the CLI entry point like:
```bash
>>> 14_cube_apply_IFFT /path/to/cube.nc --params_netcdf /path/to/config.yml [optional parameters]
```
Alternatively, the script can be executed using the (more verbose) command:
```bash
>>> python -m pseudo_3D_interpolation.cube_apply_IFFT /path/to/cube.nc \
      --params_netcdf /path/to/config.yml [optional parameters]
```

Optionally, the following parameters can be specified:

- `--help`, `-h`: Show help.
- `--params_netcdf`: Path of netCDF parameter file (YAML format). **Required!**
- `--compute_real`: Compute FFT assuming real input and thus discarting redundant negative frequencies.
- `--verbose {LEVEL}`, `-V`: Level of output verbosity (default: `0`).
