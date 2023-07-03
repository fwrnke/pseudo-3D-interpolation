---
title: Introduction
date: 2023-05-31
hide:
  - footer
---

# Result examples

## Vertical offset corrections

**Figure 2** shows the effect of the different vertical offset correction methods applied to an example dataset of >200 profiles. The [static correction](2D/static_correction.md) mainly improves the reflection coherence of individual lines (**Figure 3**) with several acquisition artifacts remaining in the data (black arrows, **Figure 2**a, b). As the survey was conducted over multiple days, a [tide compensation](2D/tide_compensation.md) basically _detrends_ the individual profiles and improves the data quality (**Figure 2**c, d). Remaining [misties](2D/mistie_correction.md) are successfully corrected using a least-squares minimization approach (**Figure 2**e, f).

<figure markdown>
![Mistie (in time domain) for individual TOPAS profiles](./figures/Index_vertical_offsets.png){ width="800" }
    <figcaption>Figure 2: Vertical offsets (upper row) and effect on detected seafloor reflection (lower row) after static correction (a, b), tide compensation (c, d), and mistie correction (e, f), respectively.</figcaption>
</figure>

## 2D TOPAS processing

The following figure illustrate the capabilities of the first workflow stage _before_ (**Figure 3**, _upper row_) and _after_ applying all processing steps (**Figure 3**, _lower row_). Both the _full waveform_ and _envelope_ of the examplary profile section are displayed for comparison. Besides compensating the vertical offsets (_larger inset figures_), a [despiking](2D/2D_despike.md) algorithm was applied to remove abundant and random noise burst of 20 ms length (_smaller inset figures_).

<figure markdown>
![Mistie (in time domain) for individual TOPAS profiles](./figures/Index_TOPAS_processing.png){ width="700" }
    <figcaption>Figure 3: Examplary TOPAS profile displayed as unprocessed full-waveform (a) and envelope (b) as well as processed sections (c) and (d), respectively.</figcaption>
</figure>

## 3D interpolation results

The following figure illustrates (a) the sparse cube _before_ and (b) the full pseudo-3D cube _after_ [interpolation](3D/3D_cube_interpolation.md) via POCS algorithm.

<figure markdown>
![Mistie (in time domain) for individual TOPAS profiles](./figures/Index_sparse_interp_3D_edit.png){ width="800" }
    <figcaption>Figure 4: Sparse (a) and interpolated (b) pseudo-3D TOPAS cube overlain by multibeam bathymetry data.</figcaption>
</figure>