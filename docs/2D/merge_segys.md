---
title: "Step 01: Merge short SEG-Y files"
date: 2022-07-27
hide:
  - footer
---

# Merge short SEG-Y files

Utility script to merge short SEG-Y file(s) with longer ones.

## Description

Merge short SEG-Y files (here: < 2000 kB, ~50 traces) with an appropriate file recorded before or after. For this decision, the spatial distance between subsequent traces is used to determine the SEG-Y file to merge with.

## Usage

This script is designed to be used from the terminal (i.e. command line).

### Command line interface

The script can handle two different inputs:

1. datalist of files to process (e.g., `datalist.txt`)
2. directory with input files (e.g., `/input_dir`) 

There are two options to run the script. We recommend using the CLI entry point like:

```bash
>>> 01_merge_segys {datalist.txt | </directory>} [optional parameters]
```

Alternatively, the script can be executed using the (more verbose) command:

```bash
>>> python -m pseudo_3D_interpolation.merge_segys {datalist.txt | </directory>} [optional parameters]
```

Optionally, the following parameters can be specified:

- `--help`, `-h`: Show help.
- `--filename_suffix {SUFFIX}`: Filename suffix (e.g. _pad_, _static_) to filter input files. Only used if directory is specified (default: `None`)
- `--suffix {sgy}`: File suffix (default: `sgy`). Only used if directory is specified.
- `--txt_suffix {despk}`: Suffix to append to output filename (default: `merge`).
- `--filesize_kB`: Threshold filesize (in kilobyte) to determine which SEG-Y will be merged (default: `2000` kB).
- `--verbose {LEVEL}`, `-V`: Level of output verbosity (default: `0`).
