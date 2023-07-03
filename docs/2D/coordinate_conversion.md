---
title: "Step 02: Coordinate conversion"
date: 2022-07-27
hide:
  - footer
---

# Coordinate conversion

Utility script for coordinate conversion (and export) from SEG-Y file(s).

## Description

This utility script reproject (i.e. transforms) coordinates read from SEG-Y header(s) to different coordinate reference systems (CRS). The coordinate transformation is conducted using the Python package [`pyproj`](https://pyproj4.github.io/pyproj/stable/), an interface to the well-kown and established [PROJ](https://proj.org/) library.
Input and output CRS can be specified as either [EPSG codes](https://epsg.io/) (e.g., `EPSG:4326`) or [PROJ.4 strings](https://proj.org/usage/quickstart.html):

```
+proj=longlat +datum=WGS84 +no_defs
```

!!! info "Suitable coordinate reference system" 

    A projected CRS is required for most subsequent processing steps (e.g. UTM)!

## Usage

This script is designed to be used from the terminal (i.e. command line).

### Command line interface

The script can handle three different inputs:

1. single SEG-Y file (e.g., `filename.sgy`)
2. datalist of files to process (e.g., `datalist.txt`)
3. directory with input files (e.g., `/input_dir`) 

There are two options to run the script. We recommend using the CLI entry point like:

```bash
>>> 02_reproject_segy {filename.sgy | datalist.txt | </directory>} [optional parameters]
```

Alternatively, the script can be executed using the (more verbose) command:

```bash
>>> python -m pseudo_3D_interpolation.reproject_segy {filename.sgy | datalist.txt | </directory>} [optional parameters]
```

Optionally, the following parameters can be specified:

- `--help`, `-h`: Show help.
- `--crs_src`: Input coordinate reference system. Indicate using EPSG code or PROJ.4 string (e.g. "epsg:4326").
- `--crs_dst`: Output coordinate reference system. Indicate using EPSG code or PROJ.4 string (e.g. "epsg:32760").
- `--output_dir {DIR}`: Output directory (either `--inplace` or `--output_dir` are required!).
- `--inplace`: Replace input data without creating copy (either `--inplace` or `--output_dir` are required!).
- `--suffix {sgy}`: File suffix (default: `sgy`). Only used if directory is specified.
- `--filename_suffix {SUFFIX}`: Filename suffix (e.g. `pad`, `static`) to filter input files. Only used if directory is specified.
- `--txt_suffix {despk}`: Suffix to append to output filename (default: `despk`).
- `--scaler`: Output coordinate scaler (following SEG-Y specification, default: `-100`).
     - *negative*: division by absolute value
     - *positive*: multiplication by absolute value
- `--src_coords`: Byte position of input coordinates in SEG-Y file(s).
     - `source`: (73, 77)
     - `group`: (81, 85)
     - `CDP`: (181, 185)
- `--dst_coords`:  Byte position of output coordinates in SEG-Y file(s).
- `--verbose {LEVEL}`, `-V`: Level of output verbosity (default: `0`).
