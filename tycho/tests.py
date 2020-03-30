"""
Created on Sat Mar  7 08:48:27 2020

@author: SamKoebrich
"""

from collections import Counter
# --- test that no duplicate plant_id_eias in eightsixty ---
# Counter(eightsixty['plant_id_eia']).most_common(1)

# --- run two generators all the way through, compare results with pickle ---


def test_CEMS():

    CEMS_hourly_path = os.path.join('data','CEMS','processed', f"CEMS_{config.TS_FREQUENCY}_2019.pkl")
    CEMS_hourly = pd.read_pickle(CEMS_hourly_path)

    CEMS_cleaned_path = os.path.join('processed','cems_clean.pkl')
    CEMS_cleaned  = pd.read_pickle(CEMS_cleaned_path)

    # --- test for nans in clean ---
    assert CEMS_cleaned.isnull().sum().sum() == 0

    # --- test for sum of hourly gross_load_mw ---
    CEMS_hourly_grouped = CEMS_hourly.groupby('plant_id_eia').resample('A', on='datetime_utc').sum()

