"""Check availability of different packages."""

from importlib import util

scipy_enabled = util.find_spec('scipy') is not None
numba_enabled = util.find_spec('numba') is not None
pywt_enabled = util.find_spec('pywt') is not None
FFST_enabled = util.find_spec('FFST') is not None  # PyShearlets
curvelops_enabled = util.find_spec('curvelops') is not None
geopandas_enabled = util.find_spec('geopandas') is not None
tpxo_tide_prediction_enabled = util.find_spec('tpxo_tide_prediction') is not None
