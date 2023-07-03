"""Utility functions for plotting."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from .utils import pad_array
from .filter import moving_average
from .signal import freq_spectrum, rms_normalization

#%%
def trim_axes(axes, N):
    """Trim unused axes from figure."""
    axes = axes.ravel()
    for ax in axes[N:]:
        ax.remove()
    return axes[:N]


# *****************************************************************************
#                           SEISMIC SECTIONS (IMAGE)
# *****************************************************************************
def plot_seismic_image(
    data,
    dt=None,
    twt=None,
    traces=None,
    cmap='Greys',
    show_colormap=True,
    show_xaxis_labels=True,
    gain=1,
    norm=False,
    title=None,
    env=False,
    reverse=False,
    units='ms',
    label_kwargs=None,
    plot_kwargs=None,
):
    """
    Plot seismic traces of SEG-Y file as image using specified colormap and gain.

    Parameters
    ----------
    data : numpy.array
        2D array of SEG-Y trace data..
    dt : float, optional
        Sampling interval in specified units (default: `seconds`).
        The default is None.
    twt : np.array, optional
        1D array of two-way traveltimes (TWT, default: `seconds`).
        The default is None.
    traces : np.array, optional
        1D array of trace indices (default: `None`).
    cmap : str, optional
        Matplotlib-compatible string of colormap (default: `Greys`).
    gain : int, optional
        Custom gain parameter (for visualization only) (default: `1`).
    norm :
        Normalize amplitude of trace(s) using `rms` or `peak` amplitude.
    title : str
        Figure title (e.g. filename) (default: `None`).
    env : bool, optional
        Envelope as input data type (default: `False`, i.e. expecting `amplitude` date).
    reverse : bool, optional
        Reverse profile orientation for plotting (default: `False`).
    units : str, optional
        Time units (y-axis) (default: `ms`).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.
    ax : matplotlib.axes.Axes
        Axes handle.
    colormap : matplotlib.colorbar.Colorbar
        Colormap handle.

    """
    # get samples and traces from data
    nsamples, ntraces = data.shape

    # create time axis (convert dt fro ms to s)
    if dt is None and twt is None:
        raise ValueError('Either dt or twt required')
    elif dt is not None:
        twt = np.linspace(0, dt * nsamples, nsamples)
    elif twt is not None:
        dt = np.mean(np.diff(twt))

    # normalize
    if norm is True or isinstance(norm, str) and norm.lower() == 'rms':
        data = rms_normalization(data, axis=0)
    elif isinstance(norm, str) and norm.lower() in ['max', 'peak']:
        data /= np.max(np.abs(data))

    # set plotting extent [xmin, xmax, ymin, ymax]
    _offset = 0.5 * dt
    extent = [-_offset, ntraces + _offset, twt[-1] + _offset, twt[0] - _offset]

    # clip amplitude data for plotting
    if gain is not None:
        clip_percentile = ((1 - gain) * 2) + 97.5  # empirically tested
        vm = np.percentile(data, clip_percentile)  # clipping
    else:
        vm = np.max(np.abs(data))
    # adjust parameter for colormap
    vmax = vm
    if env:
        vmin = 0
        data_label = 'envelope'
    else:
        vmin = -vm
        data_label = 'amplitude'

    x_label = 'trace #' if traces is None else 'field record number'
    y_label = f'time ({units})'

    if label_kwargs is None:
        label_kwargs = dict(labels_size=12, ticklabels_size=10, title_size=12)

    # create figure and axes
    if plot_kwargs is None:
        plot_kwargs = dict(figsize=(16, 8))
    fig, ax = plt.subplots(1, 1, **plot_kwargs)

    # plot data
    profile = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto', extent=extent)
    # create colormap
    if show_colormap:
        colormap = fig.colorbar(
            profile,
            ax=ax,
            pad=0.025,
            fraction=0.05,  # pad=0.01
            location='right',
            orientation='vertical',
            format='%.3f',
        )
        colormap.ax.set_ylabel(
            data_label, labelpad=25, rotation=270, fontsize=label_kwargs['labels_size']
        )
        colormap.ax.tick_params(axis='y', labelsize=label_kwargs['ticklabels_size'])

    # set x-axis
    ## ticks
    if traces is not None:
        if ntraces < 25:
            xticks = np.arange(0, ntraces, 1)
            xticklabels = [str(t) for t in traces]
        else:  # too many labels to plot for every trace
            xticks = np.arange(
                0, ntraces + 1, np.around(ntraces // 10, 1 - len(str(ntraces // 10)))
            )
            xticks = np.append(xticks, np.atleast_1d(ntraces - 1), axis=0)
            xticks = xticks[xticks < ntraces]
            xticklabels = [str(t) for t in traces[xticks]]
        ax.set_xticks(xticks)
        ax.set_xticklabels(
            xticklabels, rotation=45, ha='left', fontsize=label_kwargs['ticklabels_size']
        )
    ax.xaxis.tick_top()

    ## labels
    if show_xaxis_labels:
        ax.set_xlabel(x_label, fontweight='semibold', fontsize=label_kwargs['labels_size'])
        ax.xaxis.set_label_position('top')
    else:
        ax.set_xticklabels([])

    # set y-axis
    ## ticks
    # ax.set_ylim([twt.max(), twt.min()])
    ax.tick_params(
        axis='y', which='minor', direction='out', bottom=False, top=False, left=True, right=False
    )
    ax.yaxis.set_minor_locator(AutoMinorLocator(11))
    ## labels
    ax.set_ylabel(y_label, fontweight='semibold', fontsize=label_kwargs['labels_size'])

    ax.tick_params(axis='both', labelsize=label_kwargs['ticklabels_size'])

    # set subplot title
    if title is not None:
        ax.set_title(title, fontweight='semibold', fontsize=label_kwargs['title_size'])

    # reverse profile plot if needed
    if reverse:
        ax.invert_xaxis()

    fig.tight_layout(pad=1.1)

    if show_colormap:
        return fig, ax, colormap
    else:
        return fig, ax


def plot_seismic_image_diff(
    data_org,
    data_edit,
    dt=None,
    twt=None,
    traces=None,
    cmap='Greys',
    show_colormap=True,
    gain=1,
    env=False,
    norm=False,
    reverse=False,
    titles=None,
    units='ms',
    plot_kwargs=None,
):
    """
    Plot seismic traces of SEG-Y file as image using specified colormap and gain.

    Parameters
    ----------
    data_org, data_edit : numpy.ndarray
        2D arrays of SEG-Y trace data (original and edited).
    dt : float, optional
        Sampling interval in specified units (default: `seconds`).
        The default is None.
    twt : np.array, optional
        1D array of two-way traveltimes (TWT, default: `seconds`).
        The default is None.
    traces : np.array, optional
        1D array of trace indices (default: `None`).
    cmap : str, optional
        Matplotlib-compatible string of colormap (default: `Greys`).
    gain : int, optional
        Custom gain parameter (for visualization only) (default: `1`).
    norm :
        Normalize amplitude of trace(s) using `rms` or `peak` amplitude.
    env : bool, optional
        Envelope as input data type. The default is False (amplitude).
    reverse : bool, optional
        Reverse profile orientation for plotting (default: `False`).
    titles : list, optional
        List of plot titles (as strings). Should be exactly 3 elements.
    units : str, optional
        Time units (y-axis) (default: `ms`).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.
    ax : matplotlib.axes.Axes
        Axes handle.
    colormap : matplotlib.colorbar.Colorbar
        Colormap handle.

    """
    # get samples and traces from data
    nsamples, ntraces = data_org.shape

    # create time axis (convert dt fro ms to s)
    if dt is None and twt is None:
        raise ValueError('Either dt or twt required')
    elif dt is not None:
        twt = np.linspace(0, dt * nsamples, nsamples)
    elif twt is not None:
        dt = np.mean(np.diff(twt))

    # normalize
    if norm is True or isinstance(norm, str) and norm.lower() == 'rms':
        data_org = rms_normalization(data_org, axis=0)
        data_edit = rms_normalization(data_edit, axis=0)
    elif isinstance(norm, str) and norm.lower() in ['max', 'peak']:
        data_org /= np.max(np.abs(data_org))
        data_edit /= np.max(np.abs(data_edit))

    # create difference array
    if data_org.shape == data_edit.shape:
        data_diff = data_org - data_edit
    else:
        data_diff = np.zeros_like(data_org)

    # set plotting extent [xmin, xmax, ymin, ymax]
    _offset = 0.5 * dt
    extent = [-_offset, ntraces + _offset, twt[-1] + _offset, twt[0] - _offset]

    # clip amplitude data for plotting
    if gain is not None:
        clip_percentile = ((1 - gain) * 2) + 97.5  # empirically tested
        vm = np.percentile(data_org, clip_percentile)  # clipping
    else:
        vm = np.max(np.abs(data_org))
    # adjust parameter for colormap
    vmax = vm
    vmax_diff = np.max(np.abs(data_diff))
    if env:
        vmin = 0
        vmin_diff = 0
        data_label = 'envelope'
    else:
        vmin = -vm
        vmin_diff = -vmax_diff
        data_label = 'amplitude'

    titles = ['original', 'edited', 'difference'] if titles is None else titles
    if len(titles) != 3:
        raise ValueError('Number of elements in `titles` must be 3 but is {len(titles)}')
    x_label = 'trace #' if traces is None else 'field record number'
    y_label = f'time [{units}]'
    cmaps = [cmap, cmap, 'seismic'] if not env else [cmap, cmap, 'Reds']
    vmins = [vmin, vmin, vmin_diff]
    vmaxs = [vmax, vmax, vmax_diff]

    # create figure and axes
    if plot_kwargs is None:
        plot_kwargs = dict(figsize=(16, 8))
    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, **plot_kwargs)

    for i, data in enumerate([data_org, data_edit, data_diff]):
        # plot data
        profile = ax[i].imshow(
            data, cmap=cmaps[i], vmin=vmins[i], vmax=vmaxs[i], aspect='auto', extent=extent
        )
        # create colormap
        if show_colormap:
            colormap = fig.colorbar(
                profile,
                ax=ax[i],
                pad=0.02,
                fraction=0.05,
                shrink=0.9,
                location='bottom',
                orientation='horizontal',
                format='%.3f',
            )
            colormap.ax.set_xlabel(data_label, labelpad=5, fontsize=10)

        # set x-axis
        ## ticks

        if traces is not None:
            if ntraces < 25:
                xticks = np.arange(0, ntraces, 1)
                xticklabels = [str(t) for t in traces]
            else:  # too many labels to plot for every trace
                xticks = np.arange(
                    0, ntraces + 1, np.around(ntraces // 10, 1 - len(str(ntraces // 10)))
                )
                xticks = np.append(xticks, np.atleast_1d(ntraces - 1), axis=0)
                xticklabels = [str(t) for t in traces[xticks]]
            ax[i].set_xticks(xticks)
            ax[i].set_xticklabels(xticklabels, rotation=45, ha='left', fontsize=10)
        ax[i].xaxis.tick_top()
        ## labels
        ax[i].set_xlabel(x_label, fontweight='semibold')
        ax[i].xaxis.set_label_position('top')

        # set y-axis
        ## ticks
        # ax[i].set_ylim([twt.max(), twt.min()])
        ax[i].tick_params(
            axis='y',
            which='minor',
            direction='out',
            bottom=False,
            top=False,
            left=True,
            right=False,
        )
        ax[i].yaxis.set_minor_locator(AutoMinorLocator(11))

        # set subplot title
        ax[i].set_title(titles[i], fontweight='semibold', fontsize=12)

        # reverse profile plot if needed
        if reverse:
            ax[i].invert_xaxis()

    ## labels
    ax[0].set_ylabel(y_label, fontweight='semibold')

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.05)

    if show_colormap:
        return fig, ax, colormap
    else:
        return fig, ax


# *****************************************************************************
#                           SEISMIC SECTIONS (WIGGLE)
# *****************************************************************************
def plot_seismic_wiggle(
    data,
    dt=None,
    twt=None,
    traces=None,
    add_info=None,
    title=None,
    gain=1.0,
    norm=False,
    tr_step=1,
    color='k',
    units='ms',
    plot_kwargs=None,
):
    """
    Plot seismic section using wiggle traces.

    Parameters
    ----------
    data : np.array
        Seismic data (samples x traces).
    dt : float, optional
        Sampling interval in specified units (default: `seconds`).
        The default is None.
    twt : np.array, optional
        1D array of two-way traveltimes (TWT, default: `seconds`).
        The default is None.
    traces : np.array, optional
        1D array of trace indices (default: `None`).
    add_info : list of strings, optional
        Additional information (e.g. delay time) to annotate trace labels with.
        The default is None.
    title : str, optional
        Plot title string (default: `None`).
    gain : float, optional
        Gain value (default: `1.0`).
    norm :
        Normalize amplitude of trace(s) using `rms` or `peak` amplitude.
    tr_step : int, optional
        Plot every {tr_step} trace in data (default: `1`).
    color : str, optional
        Fill color for positive wiggle (default: `k`).
    units : str, optional
        Time units (y-axis) (default: `ms`).
    plot_kwargs : dict, optional
        Keyword arguments for plt.subplots call

    """
    if traces is not None and add_info is not None:
        assert (
            traces.size == add_info.size
        ), f'Additional annotations must be list of same length as traces ({traces.size})'

    # initialise plot
    if plot_kwargs is None:
        plot_kwargs = dict(figsize=(8, 8))

    fig, ax = plt.subplots(1, 1, **plot_kwargs)

    # select subsets using {tr_step}
    data = data[:, ::tr_step]
    traces = traces[::tr_step] if traces is not None else None
    add_info = add_info[::tr_step] if add_info is not None else None

    # get samples and traces from data
    nsamples, ntraces = data.shape

    # create time axis (convert dt fro ms to s)
    if dt is None and twt is None:
        raise ValueError('Either dt or twt required')
    elif dt is not None:
        t = np.linspace(0, dt * nsamples, nsamples)
    elif twt is not None:
        t = twt

    # normalize
    if norm is True or isinstance(norm, str) and norm.lower() == 'rms':
        data = rms_normalization(data, axis=0)
    elif isinstance(norm, str) and norm.lower() in ['max', 'peak']:
        data /= np.max(np.abs(data))

    # get start and end traces
    if traces is None:
        x_start, x_end = 1, ntraces + 1
    elif isinstance(traces, tuple):
        x_start, x_end = traces
    else:
        x_start, x_end = traces[0], traces[-1] + 1

    # get horizontal increment
    dx = np.around((x_end - x_start) / ntraces, 0)

    # create axes labels
    x_label = 'trace #' if traces is None else 'field record number'
    y_label = f'time [{units}]'

    # set x-axis
    ## ticks
    if traces is not None:
        if ntraces < 25:
            xticks = np.arange(x_start, x_end, tr_step)
            xticklabels = [str(t) for t in traces]
        else:  # too many labels to plot for every trace
            xticks = np.arange(
                x_start, x_end, np.around(ntraces // 10, 1 - len(str(ntraces // 10)))
            )
            xticklabels = [str(t) for t in traces[xticks - x_start]]
        # add additional text annotations (per trace)
        if add_info is not None:
            print(xticklabels)
            add_info = add_info[np.isin(traces, [int(t) for t in xticklabels]).nonzero()[0]]
            xticklabels = [f'{s}:{info}' for s, info in zip(xticklabels, add_info)]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=45, ha='left')
    ax.xaxis.tick_top()
    ax.set_xlim(x_start - 1, x_end)
    ## labels
    ax.set_xlabel(x_label, fontweight='semibold')
    ax.xaxis.set_label_position('top')

    # set y-axis
    ## ticks
    ax.set_ylim([t.max(), t.min()])
    ax.tick_params(
        axis='y', which='minor', direction='out', bottom=False, top=False, left=True, right=False
    )
    ax.yaxis.set_minor_locator(AutoMinorLocator(11))
    ## labels
    ax.set_ylabel(y_label, fontweight='semibold')

    # set title
    if title is not None:
        ax.set_title(title)

    for i, trace in enumerate(data.T):  # single trace per row with sample as col
        tr = trace * gain * dx  # scale trace and add offset
        x = x_start + i * dx  # calc x position for trace
        ax.plot(x + tr, t, 'k', lw=0.5)
        ax.fill_betweenx(t, x + tr, x, where=(tr >= 0), color=color)

    fig.tight_layout()

    return fig, ax


def plot_seismic_wiggle_diff(
    data_org,
    data_edit,
    dt=None,
    twt=None,
    traces=None,
    add_info=None,
    gain=1.0,
    norm=False,
    tr_step=1,
    color='k',
    titles=None,
    units='ms',
    plot_kwargs=None,
):
    """
    Plot seismic section using wiggle traces.

    Parameters
    ----------
    data_org, data_edit : np.array
        Seismic data arrays (samples x traces).
    dt : float, optional
        Sampling interval in specified units (default: `seconds`).
        The default is None.
    twt : np.array, optional
        1D array of two-way traveltimes (TWT, default: `seconds`).
        The default is None.
    traces : np.array, optional
        1D array of trace indices (default: `None`).
    add_info : list of strings, optional
        Additional information (e.g. delay time) to annotate trace labels with.
        The default is None.
    gain : float, optional
        Gain value (default: `1.0`).
    norm :
        Normalize amplitude of trace(s) using `rms` or `peak` amplitude.
    tr_step : int, optional
        Plot every {tr_step} trace in data (default: `1`).
    color : str, optional
        Fill color for positive wiggle (default: `k`).
    titles : list, optional
        List of plot titles (as strings). Should be exactly 3 elements.
    units : str, optional
        Time units (y-axis) (default: `ms`).
    plot_kwargs : dict, optional
        Keyword arguments for plt.subplots call

    """
    assert (
        data_org.shape == data_edit.shape
    ), f'Original array {data_org.shape} and edited array {data_edit.shape} must have identical shapes!'
    if traces is not None and add_info is not None:
        assert (
            traces.size == add_info.size
        ), f'Additional annotations must be list of same length as traces ({traces.size})'

    # select subsets using {tr_step}
    data = data_org[:, ::tr_step]
    traces = traces[::tr_step] if traces is not None else None
    add_info = add_info[::tr_step] if add_info is not None else None

    # get samples and traces from data
    nsamples, ntraces = data_org.shape

    # create time axis (convert dt fro ms to s)
    if dt is None and twt is None:
        raise ValueError('Either dt or twt required')
    elif dt is not None:
        t = np.linspace(0, dt * nsamples, nsamples)
    elif twt is not None:
        t = twt

    # get start and end traces
    if traces is None:
        x_start, x_end = 1, ntraces + 1
    elif isinstance(traces, tuple):
        x_start, x_end = traces
    else:
        x_start, x_end = traces[0], traces[-1] + 1

    # get horizontal increment
    dx = np.around((x_end - x_start) / ntraces, 0)

    # create axes labels
    x_label = 'trace #' if traces is None else 'field record number'
    y_label = f'time [{units}]'

    # normalize
    if norm is True or isinstance(norm, str) and norm.lower() == 'rms':
        data_org = rms_normalization(data_org, axis=0)
        data_edit = rms_normalization(data_edit, axis=0)
    elif isinstance(norm, str) and norm.lower() in ['max', 'peak']:
        data_org /= np.max(np.abs(data_org))
        data_edit /= np.max(np.abs(data_edit))

    # create difference array
    if data_org.shape == data_edit.shape:
        data_diff = data_org - data_edit
    else:
        data_diff = np.zeros_like(data_org)

    # initialise plot
    if plot_kwargs is None:
        plot_kwargs = dict(figsize=(16, 8))

    fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, **plot_kwargs)
    titles = ['original', 'edited', 'difference'] if titles is None else titles
    if len(titles) != 3:
        raise ValueError('Number of elements in `titles` must be 3 but is {len(titles)}')

    for data, ax, title in zip([data_org, data_edit, data_diff], axes, titles):
        # set x-axis
        ## ticks
        if traces is not None:
            if ntraces < 25:
                xticks = np.arange(x_start, x_end, tr_step)
                xticklabels = [str(t) for t in traces]
            else:  # too many labels to plot for every trace
                xticks = np.arange(
                    x_start, x_end, np.around(ntraces // 10, 1 - len(str(ntraces // 10)))
                )
                xticklabels = [str(t) for t in traces[xticks - x_start]]
            # add additional text annotations (per trace)
            if add_info is not None:
                xticklabels = [f'{s}:{info}' for s, info in zip(xticklabels, add_info)]
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, rotation=45, ha='left')
        ax.xaxis.tick_top()
        ax.set_xlim(x_start - 1, x_end)
        ## labels
        ax.set_xlabel(x_label, fontweight='semibold')
        ax.xaxis.set_label_position('top')

        # set y-axis
        ## ticks
        ax.set_ylim([t.max(), t.min()])
        ax.tick_params(
            axis='y',
            which='minor',
            direction='out',
            bottom=False,
            top=False,
            left=True,
            right=False,
        )
        ax.yaxis.set_minor_locator(AutoMinorLocator(11))

        # set subplot title
        ax.set_title(title, fontweight='semibold', fontsize=12)

        for i, trace in enumerate(data.T):  # single trace per row with sample as col
            tr = trace * gain * dx  # scale trace and add offset
            x = x_start + i * dx  # calc x position for trace
            ax.plot(x + tr, t, 'k', lw=0.5)
            ax.fill_betweenx(t, x + tr, x, where=(tr >= 0), color=color)
    ## labels
    axes[0].set_ylabel(y_label, fontweight='semibold')

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.1)

    return fig, axes


# *****************************************************************************
#                           SEISMIC SECTIONS (STATIC)
# *****************************************************************************
def _plot_seismic_wiggle_statics(
    data: list,
    dt=None,
    twt=None,
    traces=None,
    add_info=None,
    gain=1.0,
    tr_step=1,
    color='k',
    titles=None,
    units='ms',
    plot_kwargs=None,
):
    """Plot seismic wiggle section for static correction."""
    # get number of subplots (rows and cols)
    nplots = len(data)
    # print('nplots:', nplots)
    ncols = int(np.ceil(np.sqrt(nplots)))
    # print('ncols:', ncols)
    nrows = 1 if ncols == nplots else ncols - 1
    # print('nrows:', nrows)
    nrows = nrows + 1 if nplots > ncols * nrows else nrows
    # print('nrows:', nrows)

    assert all(a.shape == data[0].shape for a in data)
    data_ref = data[0]

    # select subsets using {tr_step}
    # data = data_ref[:,::tr_step]
    # traces = traces[::tr_step] if traces is not None else None
    # add_info = add_info[::tr_step] if add_info is not None else None

    if add_info is not None and not isinstance(add_info, list):
        add_info = [add_info]

    # get samples and traces from data
    nsamples, ntraces = data_ref.shape
    # print('nsamples, ntraces:', nsamples, ntraces)

    # create time axis (convert dt fro ms to s)
    if dt is None and twt is None:
        raise ValueError('Either dt or twt required')
    elif dt is not None:
        t = np.linspace(0, dt * nsamples, nsamples)
    elif twt is not None:
        t = twt

    # get start and end traces
    if traces is None:
        x_start, x_end = 1, ntraces + 1
    elif isinstance(traces, tuple):
        x_start, x_end = traces
    else:
        x_start, x_end = traces[0], traces[-1] + 1

    # get horizontal increment
    dx = np.around((x_end - x_start) / ntraces, 0)

    # create axes labels
    x_label = 'trace #' if traces is None else 'field record number'
    y_label = f'time [{units}]'

    if plot_kwargs is None:
        fig_kwargs = dict(figsize=(ncols * 4, nrows * 3))

    # initialize figure
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, **fig_kwargs)

    # set labels
    if nplots > 1:
        for xax in axes[0, :]:
            xax.set_xlabel(x_label, fontweight='semibold')
            xax.xaxis.set_label_position('top')
        for yax in axes[:, 0]:
            yax.set_ylabel(y_label, fontweight='semibold')
    else:
        axes.set_xlabel(x_label, fontweight='semibold')
        axes.xaxis.set_label_position('top')
        axes.set_ylabel(y_label, fontweight='semibold')

    # prepare subplot axes (remove unused subplots, account for nplots ==1)
    axes = trim_axes(axes, nplots) if nplots > 1 else axes
    axes_iter = [axes] if ntraces == 1 else axes

    # loop over every trace in input array
    for i, ax in enumerate(axes_iter):
        # set x-axis
        if i < ncols:  # add xticklabels at top
            ax.tick_params(axis='x', which='both', labeltop=True)
        else:
            ax.tick_params(axis='x', which='both', labeltop=False)
        ## ticks
        if traces is not None:
            if ntraces < 25:
                xticks = np.arange(x_start, x_end, tr_step)
                xticklabels = [str(t) for t in traces]
            else:  # too many labels to plot for every trace
                xticks = np.arange(
                    x_start, x_end, np.around(ntraces // 10, 1 - len(str(ntraces // 10)))
                )
                xticklabels = [str(t) for t in traces[xticks - x_start]]

            # add additional text annotations (per trace)
            if add_info is not None:
                xticklabels = [f'{s}:{info}' for s, info in zip(xticklabels, add_info[i])]

            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, rotation=45, ha='left')

        ax.xaxis.tick_top()
        ax.set_xlim(x_start - 1, x_end)

        if i < ncols:
            ax.tick_params(axis='x', which='both', labeltop=True)
        else:
            ax.tick_params(axis='x', which='both', labeltop=False)

        # set y-axis
        ## ticks
        ax.set_ylim([t.max(), t.min()])
        ax.tick_params(
            axis='y',
            which='minor',
            direction='out',
            bottom=False,
            top=False,
            left=True,
            right=False,
        )
        ax.yaxis.set_minor_locator(AutoMinorLocator(11))

        # set subplot title
        if titles is not None:
            # ax.set_title(titles[i], fontweight='semibold', fontsize=12, pad=-6)
            ax.text(
                0.5,
                0.98,
                titles[i],
                fontweight='semibold',
                fontsize=11,
                # backgroundcolor='white',
                bbox=dict(facecolor='white', alpha=0.7, lw=None),
                transform=ax.transAxes,
                horizontalalignment='center',
                verticalalignment='top',
            )

        for i, trace in enumerate(data[i].T):  # single trace per row with sample as col
            tr = trace * gain * dx  # scale trace and add offset
            x = x_start + i * dx  # calc x position for trace
            ax.plot(x + tr, t, 'k', lw=0.5)
            ax.fill_betweenx(t, x + tr, x, where=(tr >= 0), color=color)

    return fig, axes


# *****************************************************************************
#                           FREQUENCY SPECTRA
# *****************************************************************************
def plot_trace_freq_spectrum(
    data,
    dt=None,
    Fs=None,
    trace_labels=None,
    units='ms',
    plot_mvg_avg=True,
    plot_combined=True,
    fig_kwargs=None,
):
    """
    Plot frequency spectrum of input trace(s).

    Parameters
    ----------
    data : np.ndarray
        Trace data (samples x traces).
    dt : float, optional
        Sampling interval in milliseconds [ms]. Either `dt` or `Fs` needed.
    Fs : int, optional
        Sampling rate [Hz]. Either `dt` or `Fs` needed.
    trace_labels : np.ndarray, optional
        Array of trace labels (e.g. field record numbers) (default: `None`).
    units : str, optional
        Unit of `dt` as ['s', 'ms', 'ns'] (default: `ms`).
    plot_mvg_avg : bool, optional
        Plot moving average of spectrum(s) (default: `True`).
    plot_combined : bool, optional
        Plot average spectrum of all traces (default: `True`).
    fig_kwargs : dict, optional
        Optional keyword argument for figure creation (default: `None`).

    Returns
    -------
    fig : figure.Figure
        Matplotlib igure object.
    ax : axes.Subplots
        Matplotlib subplot axes.

    """
    # get number of samples and traces
    if data.ndim == 1:
        nsamples, ntraces = data.size, 1
        ncols, nrows = 1, 1
        data = np.atleast_2d(data).T
    else:
        nsamples, ntraces = data.shape
        print('data.shape:', data.shape)
        ntr = ntraces + 1 if plot_combined and ntraces > 1 else ntraces
        print('ntr:', ntr)
        ncols = int(np.ceil(np.sqrt(ntr)))
        print('ncols:', ncols)
        nrows = 1 if ncols == ntr else ncols - 1
        print('nrows:', nrows)
        nrows = nrows + 1 if ntr > ncols * nrows else nrows
        print('nrows:', nrows)

    if dt is None and Fs is None:
        raise ValueError('Either `dt` or `Fs` is required.')
    if dt is not None and Fs is None:
        if units == 's':
            pass
        elif units == 'ms':
            dt = dt / 1000
        elif units == 'ns':
            dt = dt / 1e-6
        else:
            raise ValueError(f'Unit "{units}" is not supported for `dt`.')

        Fs = 1 / dt  # in Hz (samples/s)

    if plot_mvg_avg:
        avg_win_size = int(nsamples * 0.01)  # 1% of nsamples
        avg_win_size = 2 if avg_win_size < 2 else avg_win_size
        print('avg_win_size:', avg_win_size)

    if fig_kwargs is None:
        fig_kwargs = dict(figsize=(ncols * 4, nrows * 3))

    amps, fpeaks, fmins, fmaxs = [], [], [], []

    # initialize figure
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, **fig_kwargs)

    # set labels
    if ntraces > 1:
        for xax in axes[-1, :]:
            xax.set_xlabel('frequency [Hz]', fontsize=11)
        for yax in axes[:, 0]:
            yax.set_ylabel('amplitude [-]', fontsize=11)
    else:
        axes.set_xlabel('frequency [Hz]', fontsize=11)
        axes.set_ylabel('amplitude [-]', fontsize=11)

    # prepare subplot axes (remove unused subplots, account for ntraces ==1)
    axes = trim_axes(axes, ntr) if ntraces > 1 else axes
    axes_iter = axes[:-1] if plot_combined and ntraces > 1 else axes
    axes_iter = [axes_iter] if ntraces == 1 else axes_iter

    # loop over every trace in input array
    for i, ax in enumerate(axes_iter):
        # compute frequency spectrum
        f, amp, f_min, f_max = freq_spectrum(data[:, i], Fs, return_minmax=True)
        # get frequency of peak amplitude
        f_peak = f[np.argmax(amp)]

        # plot frequency spectrum
        ax.plot(f, amp, c='b', lw=0.5)

        # plot moving window average
        if plot_mvg_avg:
            # amp_win = moving_average(amp, avg_win_size, pad=True)
            half_window = (avg_win_size - 1) // 2
            amp_win = pad_array(amp, half_window, zeros=True)
            amp_win = moving_average(amp_win, win=avg_win_size)
            fx = f if f.size == amp_win.size else f[:-1]
            ax.plot(fx, amp_win, c='k', lw=1)

        # set title
        title = f'trace #{trace_labels[i]}' if trace_labels is not None else f'trace #{i}'
        ax.set_title(title, fontsize=12, fontweight='semibold')

        # annotate plot
        if f_min > 1000:
            stats = f'\nMin: {f_min/1000:.1f} kHz\nMax: {f_max/1000:.1f} kHz\nPeak: {f_peak/1000:.1f} kHz'
        else:
            stats = f'\nMin: {f_min:.1f} Hz\nMax: {f_max:.1f} Hz\nPeak: {f_peak:.1f} Hz'
        ax.text(
            0.98,
            0.95,
            'AMPLITUDE SPECTRUM',
            horizontalalignment='right',
            verticalalignment='top',
            fontweight='normal',
            color='b',
            transform=ax.transAxes,
            fontsize=11,
        )
        ax.text(
            0.98,
            0.95,
            stats,
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax.transAxes,
            fontsize=9,
        )

        # append trace spectrum parameter to list
        amps.append(amp)
        fpeaks.append(f_peak)
        fmins.append(f_min)
        fmaxs.append(f_max)

    # set overall x-axis limits
    # ax.set_xlim([np.min(fmins), np.max(fmaxs)])

    if plot_combined and ntraces > 1:
        # calc averages
        amp_mean = np.nanmean(np.array(amps), axis=0)
        fpeaks_mean = np.mean(fpeaks)
        fmins_mean = np.mean(fmins)
        fmaxs_mean = np.mean(fmaxs)
        f_nyquist = Fs // 2

        # plot average frequency spectrum
        axes[-1].fill_between(f, 0, amp_mean, color='grey')

        # plot moving window average
        # amp_mean_win = moving_average(amp_mean, avg_win_size, pad=True)
        half_window = (avg_win_size - 1) // 2
        amp_mean_win = pad_array(amp_mean, half_window, zeros=True)
        amp_mean_win = moving_average(amp_mean_win, win=avg_win_size)
        fx = f if f.size == amp_mean_win.size else f[:-1]
        axes[-1].plot(fx, amp_mean_win, 'k', lw=1)

        # annotate plot
        axes[-1].set_title('average spectrum', fontsize=12, fontweight='semibold')
        if f_min > 1000:
            stats = f'\nMin: {fmins_mean/1000:.1f} kHz\nMax: {fmaxs_mean/1000:.1f} kHz\nPeak: {fpeaks_mean/1000:.1f} kHz\nNyquist: {f_nyquist/1000:.1f} kHz'
        else:
            stats = f'\nMin: {fmins_mean:.1f} Hz\nMax: {fmaxs_mean:.1f} Hz\nPeak: {fpeaks_mean:.1f} Hz\nNyquist: {f_nyquist:.1f} Hz'
        axes[-1].text(
            0.98,
            0.95,
            'AMPLITUDE SPECTRUM',
            horizontalalignment='right',
            verticalalignment='top',
            fontweight='bold',
            color='k',
            transform=axes[-1].transAxes,
        )
        axes[-1].text(
            0.98,
            0.95,
            stats,
            horizontalalignment='right',
            verticalalignment='top',
            transform=axes[-1].transAxes,
            fontsize=9,
        )
    return fig, ax


def plot_average_freq_spectrum(
    data, dt=None, Fs=None, trace_labels=None, plot_mvg_avg=True, fig_kwargs=None
):
    """
    Plot frequency spectrum of input trace(s).

    Parameters
    ----------
    data : np.ndarray
        Trace data (samples x traces).
    dt : float, optional
        Sampling interval in milliseconds [ms]. Either `dt` or `Fs` needed.
    Fs : int, optional
        Sampling rate [Hz]. Either `dt` or `Fs` needed.
    trace_labels : np.ndarray, optional
        Array of trace labels (e.g. field record numbers) (default: `None`).
    plot_mvg_avg : bool, optional
        Plot moving average of spectrum(s) (default: `True`).
    fig_kwargs : dict, optional
        Optional keyword argument for figure creation (default: `None`).

    Returns
    -------
    fig : figure.Figure
        Matplotlib igure object.
    ax : axes.Subplots
        Matplotlib subplot axes.

    """
    # get number of samples and traces
    if data.ndim == 1:
        nsamples, ntraces = data.size, 1
        data = np.atleast_2d(data).T
    else:
        nsamples, ntraces = data.shape

    if dt is None and Fs is None:
        raise ValueError('Either `dt` or `Fs` is required.')
    if dt is not None and Fs is None:
        dt = dt / 1000  # convert ms to s
        Fs = 1 / dt  # in Hz (samples/s)

    if plot_mvg_avg:
        avg_win_size = int(nsamples * 0.01)  # 1% of nsamples
        avg_win_size = 2 if avg_win_size < 2 else avg_win_size
        print('avg_win_size:', avg_win_size)

    if fig_kwargs is None:
        fig_kwargs = dict(figsize=(12, 8))

    amps, fpeaks, fmins, fmaxs = [], [], [], []

    # initialize figure
    fig, ax = plt.subplots(nrows=1, ncols=1, **fig_kwargs)

    # loop over every trace in input array
    for i in range(ntraces):
        # compute frequency spectrum
        f, amp, f_min, f_max = freq_spectrum(data[:, i], Fs, return_minmax=True)
        # get frequency of peak amplitude
        f_peak = f[np.argmax(amp)]

        # append trace spectrum parameter to list
        amps.append(amp)
        fpeaks.append(f_peak)
        fmins.append(f_min)
        fmaxs.append(f_max)

    # set overall x-axis limits
    # ax.set_xlim([np.min(fmins), np.max(fmaxs)])
    print(np.min(fmins), np.max(fmaxs))
    # set labels
    ax.set_xlabel('frequency [Hz]', fontsize=11)
    ax.set_ylabel('amplitude [-]', fontsize=11)

    # calc averages
    amp_mean = np.nanmean(np.array(amps), axis=0)
    fpeaks_mean = np.mean(fpeaks)
    fmins_mean = np.mean(fmins)
    fmaxs_mean = np.mean(fmaxs)
    f_nyquist = Fs // 2

    # plot average frequency spectrum
    ax.fill_between(f, 0, amp_mean, color='grey')

    # plot moving window average
    half_window = (avg_win_size - 1) // 2
    amp_mean_win = pad_array(amp_mean, half_window, zeros=True)
    amp_mean_win = moving_average(amp_mean_win, win=avg_win_size)
    fx = f if f.size == amp_mean_win.size else f[:-1]
    ax.plot(fx, amp_mean_win, 'k', lw=1)

    # annotate plot
    ax.set_title('average spectrum', fontsize=12, fontweight='semibold')
    if f_min > 1000:
        stats = f'\nMin: {fmins_mean/1000:.1f} kHz\nMax: {fmaxs_mean/1000:.1f} kHz\nPeak: {fpeaks_mean/1000:.1f} kHz\nNyquist: {f_nyquist/1000:.1f} kHz'
    else:
        stats = f'\nMin: {fmins_mean:.1f} Hz\nMax: {fmaxs_mean:.1f} Hz\nPeak: {fpeaks_mean:.1f} Hz\nNyquist: {f_nyquist:.1f} Hz'
    ax.text(
        0.98,
        0.95,
        'AMPLITUDE SPECTRUM',
        horizontalalignment='right',
        verticalalignment='top',
        fontweight='bold',
        color='k',
        transform=ax.transAxes,
    )
    ax.text(
        0.98,
        0.95,
        stats,
        horizontalalignment='right',
        verticalalignment='top',
        transform=ax.transAxes,
        fontsize=9,
    )
    return fig, ax
