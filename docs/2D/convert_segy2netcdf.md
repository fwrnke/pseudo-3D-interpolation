---
title: "Step 09: Convert SEG-Y to netCDF"
date: 2022-07-27
hide:
  - footer
---

# Convert SEG-Y files to netCDF format

## Description

This utility script converts (multiple) SEG-Y file(s) to netCDF format using the Python package [`segysak`](https://segysak.readthedocs.io/en/latest/) in parallel.

## Usage

This script is designed to be used from the terminal (i.e. command line).

### Command line interface

The script can handle two different inputs:

1. single SEG-Y file (e.g., `filename.sgy`)
2. datalist of files to process (e.g., `datalist.txt`)
3. directory with input files (e.g., `/input_dir`) 

There are two options to run the script. We recommend using the CLI entry point like:

```bash
>>> 09_convert_segy2netcdf {filename.sgy | datalist.txt | </directory>} [optional parameters]
```

Alternatively, the script can be executed using the (more verbose) command:

```bash
>>> python -m pseudo_3D_interpolation.cnv_segy2netcdf {filename.sgy | datalist.txt | </directory>} [optional parameters]
```

Optionally, the following parameters can be specified:

- `--help`, `-h`: Show help.
- `--output_dir {DIR}`: Output directory.
- `--suffix {sgy}`: File suffix (default: `sgy`). Only used if directory is specified.
- `--filename_suffix {SUFFIX}`: Filename suffix (e.g. `pad`, `static`) to filter input files. Only used if directory is specified.
- `--nprocesses`: Number of parallel conversions (should be ≤ number of available CPUs).
- `--verbose {LEVEL}`, `-V`: Level of output verbosity (default: `0`).
