---
title: "Step 15: netCDF to SEG-Y conversion"
date: 2022-08-12
hide:
  - footer
---

# netCDF to SEG-Y conversion
This utility script converts pseudo-3D cube from netCDF-4 to SEG-Y format.

## Description
The format conversion is done by utilizing the [`SEGY-SAK`](https://segysak.readthedocs.io/en/latest/index.html)  together with the [`segyio`](https://segyio.readthedocs.io/en/latest/index.html) packages.

### Command line interface
The script accepts a **netCDF cube** (`*.nc`) as input:

There are two options to run the script. We recommend using the CLI entry point like:
```bash
>>> 15_cube_cnv_netcdf2segy /path/to/cube.nc --params_netcdf /path/to/config.yml [optional parameters]
```
Alternatively, the script can be executed using the (more verbose) command:
```bash
>>> python -m pseudo_3D_interpolation.cube_cnv_netcdf2segy_3D /path/to/cube.nc \
      --params_netcdf /path/to/config.yml [optional parameters]
```

Optionally, the following parameters can be specified:

- `--help`, `-h`: Show help.
- `--params_netcdf`: Path of netCDF parameter file (YAML format). **Required!**
- `--path_segy`: Path of output SEG-Y file. Defaults to using input netCDF filename.
- `--scalar_coords`: Coordinate scalar for SEG-Y trace header [100, 10, 1, -10, -100, 'auto']. Defaults to `auto`, i.e. applying a suitable scalar derived from the coordinates stored in the netCDF.
- `--verbose {LEVEL}`, `-V`: Level of output verbosity (default: `0`).