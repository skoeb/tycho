
# --- Python Batteries Included---
import os

# --- External Libraries ---
import pandas as pd
import numpy as np

# --- Module Imports ---
import tycho
from tycho.config import *

import logging
log = logging.getLogger("tycho")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~ Read Pickles ~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
merged = pd.read_pickle(os.path.join('processed', 'merged_df.pkl'))
gppd = pd.read_pickle(os.path.join('processed','gppd_clean.pkl'))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~ Plot ~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

tycho.plot_cems_emissions(merged)
tycho.plot_corr_heatmap(merged)
tycho.plot_eda_pair(merged)

# --- Plot training data ---
tycho.plot_map_plants(merged,
                      country='United States of America',
                      title='U.S. Power Plants Included in Tycho Training Set')

tycho.plot_map_plants(gppd,
                      country=PREDICT_COUNTRIES[0],
                      tiles=10)

# --- Plot Prediction ---
tycho.plot_emission_factor(data_type='test', country='U.S.')