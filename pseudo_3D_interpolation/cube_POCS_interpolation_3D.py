"""
Interpolating sparse 3D volume using iterative POCS algorithm.
The following **sparse transforms** are available:
  
  - `FFT`      (provided by `numpy.fft`)
  - `WAVELET`  (provided by `PyWavelets`)
  - `SHEARLET` (based on custom fork of `PyShearlets` package with **disabled multithreading**)
  - `CURVELET` (**Unix** systems **only**!)

!!! warning

    The `CURVELET` transform from  is only available on Unix systems as it relies on `FFTW` version 2.1.5
    that is **obsolete** and was last released in 1999. For further instruction on how to install
    the Python interface `curvelops` (build on top of `pylops`) please refer to their documentation.
    
References
----------
[^1]: NumPy ([GitHub](https://github.com/numpy/numpy))
[^2]: PyWavelets ([GitHub](https://github.com/PyWavelets/pywt))
[^3]: PyShearlets ([GitHub](https://github.com/fwrnke/PyShearlets))
[^4]: PyLops ([GitHub](https://github.com/PyLops/pylops))
[^5]: curvelops ([GitHub](https://github.com/PyLops/curvelops))

"""
import os
import sys
import glob
import yaml
import argparse
import datetime
import itertools
from functools import partial
import warnings

import numpy as np
import xarray as xr
import dask
from dask.distributed import Client, LocalCluster, performance_report, progress
from dask.diagnostics import ProgressBar

from pseudo_3D_interpolation.functions.POCS import POCS, FPOCS, APOCS
from pseudo_3D_interpolation.functions.utils import xprint
from pseudo_3D_interpolation.functions.backends import pywt_enabled, FFST_enabled, curvelops_enabled

if pywt_enabled:
    import pywt
else:
    warnings.warn('Module `pywavelet` not found. WAVELET transform is not available.')
    
if FFST_enabled:
    import FFST
else:
    warnings.warn('Module `pyshearlets` not found. SHEARLET transform is not available.')

if curvelops_enabled:
    import curvelops
else:
    warnings.warn('Module `curvelops` not found. CURVELET transform is not available.')

xr.set_options(keep_attrs=True)  # preserve metadata during computations

POCS_VERSIONS = {'POCS': POCS, 'FPOCS': FPOCS, 'APOCS': APOCS}

# %% FUNCTIONS
ffloat = partial(np.format_float_positional, trim='-')


def define_input_args():  # noqa
    parser = argparse.ArgumentParser(description='Interpolate sparse 3D cube using POCS algorithm.')
    parser.add_argument(
        'path_cube', type=str, help='Input path of 3D cube'
    )
    parser.add_argument(
        '--path_pocs_parameter', type=str, required=True,
        help='Path of netCDF parameter file (YAML format).'
    )
    parser.add_argument(
        '--path_output_dir', type=str, help='Output directory for interpolated slices.'
    )
    parser.add_argument(
        "--verbose", "-V", type=int, nargs="?", default=0, choices=[0, 1, 2],
        help="Level of output verbosity (default: 0)",
    )
    return parser


def split_by_chunks(dataset):
    """
    Split `xarray.Dataset` into sub-datasets (for netCDF export).

    Parameters
    ----------
    dataset : xr.Dataset
        Input dataset to split.

    Yields
    ------
    generator (of xr.Datasets)
        Sub-datasets of input dataset.

    References
    ----------
    [^1]: [https://ncar.github.io/esds/posts/2020/writing-multiple-netcdf-files-in-parallel-with-xarray-and-dask/#create-a-helper-function-to-split-a-dataset-into-sub-datasets](https://ncar.github.io/esds/posts/2020/writing-multiple-netcdf-files-in-parallel-with-xarray-and-dask/#create-a-helper-function-to-split-a-dataset-into-sub-datasets)

    """
    chunk_slices = {}
    for dim, chunks in dataset.chunks.items():
        slices = []
        start = 0
        for chunk in chunks:
            if start >= dataset.sizes[dim]:
                break
            stop = start + chunk
            slices.append(slice(start, stop))
            start = stop
        chunk_slices[dim] = slices
    for slices in itertools.product(*chunk_slices.values()):
        selection = dict(zip(chunk_slices.keys(), slices))
        yield dataset[selection]


def dataset_subsets(dataset, dim: str, size: int):
    """
    Generate dataset views of given `size` along `dim`.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset to subset.
    dim : str
        Dimension along which to subset.
    size : int
        Size of subset along `dim`.

    Yields
    ------
    xr.Dataset
        Subset of input dataset.

    """
    indices = list(range(0, dataset[dim].size + size, size))
    for start, end in zip(indices[:-1], indices[1:]):
        yield dataset[{dim: slice(start, end)}]


def create_file_path(ds, dim='twt', prefix=None, root_path='.'):
    """Generate a file path when given an `xarray.Dataset`."""
    if prefix is None:
        prefix = datetime.datetime.today().strftime('%Y-%m-%d')
    try:
        start = ds[dim].data[0]
        end = ds[dim].data[-1]  # noqa
    except IndexError:
        start = np.atleast_1d(ds[dim].data)[0]
        end = np.atleast_1d(ds[dim].data)[-1]  # noqa

    return os.path.join(root_path, f'{prefix}_{start:06.3f}_{end:06.3f}.nc')


def split_complex_variable(ds, var, out_dtype='float32'):
    """Split complex `xr.DataArray` variable (`var`) into _Real_ and _Imag_ parts."""
    ds[f'{var}.real'] = ds[var].real
    ds[f'{var}.imag'] = ds[var].imag
    return ds.drop(var)


# def gaussian_filter_rescale(data: np.ndarray, sigma=1) -> np.ndarray:
#     """Filter and rescale 2D array using Gaussian function."""
#     return rescale(gaussian_filter(data, sigma=sigma), vmin=data.min(), vmax=data.max())


# def median_filter_rescale(data: np.ndarray, size=(3, 3)) -> np.ndarray:
#     """Filter and rescale 2D array using median function."""
#     return rescale(median_filter(data, size=size), vmin=data.min(), vmax=data.max())


def combine_runtime_results(dir_files: str, prefix: str = 'combined', fsuffix: str = 'out') -> None:
    """
    Combine individual runtime result files into single file.

    Parameters
    ----------
    dir_files : str
        File directory.
    prefix : str, optional
        Filename prefix (default: `combined`).
    fsuffix : str, optional
        File suffix of ouput runtime results (default: `out`).

    """
    files = glob.glob(os.path.join(dir_files, f'*.{fsuffix}'))
    with open(os.path.join(dir_files, f'runtimes_{prefix}.txt'), mode='w', newline='\n') as fout:
        for file in files:
            with open(file, mode='r') as f:
                fout.write(f.read())


def main(argv=sys.argv):
    """Interpolate sparse 3D cube."""
    SCRIPT = os.path.basename(__file__)
    TODAY = datetime.date.today().strftime('%Y-%m-%d')
    
    parser = define_input_args()
    args = parser.parse_args(argv[1:])  # exclude filename parameter at position 0
    
    verbose = args.verbose

    # parameter
    xprint("Load POCS parameter from config file", kind="info", verbosity=verbose)
    with open(args.path_pocs_parameter, mode="r") as f:
        cfg = yaml.safe_load(f)
        cfg['metadata']['transform_kind'] = cfg['metadata']['transform_kind'].upper()
    
    metadata = cfg['metadata']
    TRANSFORM = metadata['transform_kind']

    # input 3D cube
    path_cube = args.path_cube
    dir_work, file = os.path.split(path_cube)
    filename, suffix = os.path.splitext(file)

    # output folder (for interpolated chunks)
    prefix = f"{filename}_{TRANSFORM}_{metadata['thresh_op']}_niter-{metadata['niter']}"
    out_path = (
        args.path_output_dir if args.path_output_dir is not None else os.path.join(dir_work, prefix)
    )
    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    # open dataset
    dim = cfg['dim']  # 'freq_twt' if 'freq' in file else 'twt'
    chunks = {dim: 1, 'iline': -1, 'xline': -1}  # dim=1 --> **faster** than dim=20!!
    cube = xr.open_dataset(path_cube, chunks=chunks, engine='h5netcdf')
    shape = tuple([v for k, v in cube.dims.items() if k in ['iline', 'xline']])
    # dim = [d for d in list(cube.dims) if d not in ['iline','xline']][0]
    var = cfg.get('var', [v for v in list(cube.data_vars) if v != 'fold'][0])

    # load fold into memory
    cube['fold'].load()

    # create data mask from fold
    cube['mask'] = cube['fold'].where(
        cube['fold'] <= 1, other=1
    )  # preserve coord attrs by using `xr.DataArray.where` instead of `xr.where`

    # check for complex input (frequency domain)
    COMPLEX = np.iscomplexobj(cube[var])

    # [POCS] initialize POCS parameter
    # write parameter to disk
    with open(os.path.join(out_path, f'parameter_{prefix}.yml'), mode='w', newline='\n') as f:
        yaml.safe_dump(metadata, f)

    # create FFT transform
    if TRANSFORM == 'FFT':
        metadata['transform'] = np.fft.fft2
        metadata['itransform'] = np.fft.ifft2

    # create WAVELET transform
    elif TRANSFORM == 'WAVELET' and pywt_enabled:
        wavelet = metadata.get('wavelet', 'coif5')  # db30
        wavelet_mode = 'smooth'
        metadata['transform'] = partial(pywt.wavedec2, wavelet=wavelet, mode=wavelet_mode)
        metadata['itransform'] = partial(pywt.waverec2, wavelet=wavelet, mode=wavelet_mode)

        prefix += f'_{wavelet}-{wavelet_mode}'

    # create SHEARLET transform
    elif TRANSFORM == 'SHEARLET' and FFST_enabled:
        Psi = FFST.scalesShearsAndSpectra(
            shape, numOfScales=None, realCoefficients=True, fftshift_spectra=True
        )
        metadata['transform'] = FFST.shearletTransformSpect
        metadata['itransform'] = FFST.inverseShearletTransformSpect

    # create Curvelet transform
    elif TRANSFORM == 'CURVELET' and curvelops_enabled:
        nbangles_coarse = 20  # default: 16
        allcurvelets = True
        DCTOp = curvelops.FDCT2D(
            shape, nbscales=None, nbangles_coarse=nbangles_coarse, allcurvelets=allcurvelets
        )
        metadata['transform'] = DCTOp.matvec
        metadata['itransform'] = DCTOp.rmatvec

        prefix += f'_nbangles-{nbangles_coarse}'

    else:
        raise ValueError(f'Transform < {metadata["transform_kind"]} > is not supported.')

    # [DASK] Setup dask distributed cluster
    cluster_config = dict(
        n_workers=cfg['n_workers'],
        processes=cfg['processes'],
        threads_per_worker=cfg['threads_per_worker'],
        memory_limit=cfg['memory_limit']
    )

    # [POCS] Interpolation using `dask`
    with LocalCluster(**cluster_config, silence_logs=50) as cluster, Client(cluster) as client:

        # create slices to process
        indices = list(range(0, cube[dim].size + cfg['batch_chunk'], cfg['batch_chunk']))
        slices = [slice(start, stop) for start, stop in zip(indices[:-1], indices[1:])]

        # apply POCS for each slice
        aux = Psi if TRANSFORM == 'SHEARLET' else None  # only needed for SHEARLETS
        input_core_dims = (
            [['iline', 'xline'], ['iline', 'xline'], ['iline', 'xline', 'lvl']]
            if TRANSFORM == 'SHEARLET'
            else [['iline', 'xline'], ['iline', 'xline'], []]
        )

        ds_interp = [
            xr.apply_ufunc(
                POCS,
                cube.isel({dim: cube_slice})[var],
                cube['mask'],
                aux,
                input_core_dims=input_core_dims,
                output_core_dims=[['iline', 'xline']],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[cube[var].dtype],
                keep_attrs=True,
                kwargs=(
                    dict(
                        metadata,
                        path_results=os.path.join(
                            out_path, f"slice-{cube_slice.start:04d}-{cube_slice.stop:04d}.out"
                        ),
                    )
                    if cfg['output_runtime_results']
                    else metadata
                ),
            )
            .to_dataset(name=f'{var}_interp')
            .assign(fold=cube.fold)
            for cube_slice in slices
        ]

        if COMPLEX:
            ds_interp = [split_complex_variable(ds, var=f'{var}_interp') for ds in ds_interp]

        # add metadata attributes
        exclude_keys = ['transform', 'itransform', 'results_dict', 'path_results']

        for ds in ds_interp:
            ds.attrs = cube.attrs  # copy input global attributes
            ds['iline'].attrs = cube['iline'].attrs  # not preserved by xr.apply_ufunc (core dims)
            ds['xline'].attrs = cube['xline'].attrs  # not preserved by xr.apply_ufunc (core dims)
            attrs_domain = '(frequency domain)' if 'freq' in dim else '(time domain)'
            ds.attrs.update(
                {
                    'description': f'Interpolated pseudo-3D cube using {metadata["transform_kind"]} '
                    + 'transform created from TOPAS profiles '
                    + attrs_domain,
                    'interp_params_keys': ';'.join([k for k in metadata if k not in exclude_keys]),
                    'interp_params_vals': ';'.join(
                        [str(metadata[k]) for k in metadata if k not in exclude_keys]
                    ),
                    'history': cube.attrs.get('history', '')
                    + f'{SCRIPT}:{metadata["transform_kind"]} {attrs_domain};',
                    'text': cube.attrs.get('text', '')
                    + f'\n{TODAY}: {metadata["transform_kind"]} {attrs_domain.upper()}'
                }
            )

        # create output file paths
        paths = [
            create_file_path(ds, dim=dim, prefix=prefix, root_path=out_path) for ds in ds_interp
        ]

        out_batch = [
            ds.to_netcdf(path, engine='h5netcdf', compute=False) for ds, path in zip(ds_interp, paths)
        ]

        path_report = os.path.join(out_path, f'dask-report_{metadata["transform_kind"]}.html')
        with performance_report(filename=path_report):
            # trigger computation
            futures = client.compute(out_batch)
            # show progress bar
            progress(futures, notebook=False)

    # [DASK] finalize and close cluster
    cluster.close()

    # [RESULTS] Merge individual runtime results files into single file
    if cfg['output_runtime_results']:
        combine_runtime_results(out_path, prefix=prefix)

    # [RESULTS] Merge individual netCDF slices into single file
    xprint('Open multiple files as a single dataset', kind='info', verbosity=verbose)
    cube_fft = xr.open_mfdataset(
        os.path.join(out_path, '*.nc'),
        chunks=chunks,
        engine='h5netcdf',
        parallel=True,
        data_vars='minimal',
    )
    
    with dask.config.set(scheduler='threads'), ProgressBar():
        xprint('Write combinded netCDF file to disk', kind='info', verbosity=verbose)
        # write merged output file
        cube_fft.to_netcdf(f'{out_path}.nc', engine='h5netcdf')  # 1min 60.0s (226 files -> 4501 slices)


#%% MAIN
if __name__ == '__main__':
    
    main()    
