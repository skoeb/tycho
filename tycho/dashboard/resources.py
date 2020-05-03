
# --- Battery imports ---
import os

# --- External modules import ---
import pandas as pd
import pycountry_convert as pcc

def alpha3_to_country(alpha3):
    alpha2 = pcc.country_alpha3_to_country_alpha2(alpha3)
    country = pcc.country_alpha2_to_country_name(alpha2)

    # --- split at comma ---
    country = country.split(',')[0]
    return country

# --- Import dfs used throughout ---
long_df = pd.read_pickle(os.path.join('data','dashboard','dashboard_df.pkl'))
long_df['country'] = long_df['country'].apply(alpha3_to_country)

# --- Dict to lookup label from value ---
variable_lookup = {
    'co2_lbs': 'CO2 lbs',
    'nox_lbs': 'NOx lbs',
    'so2_lbs': 'SO2 lbs',
    'co2_lbs_ef_mwh': 'CO2 lbs/MWh',
    'nox_lbs_ef_mwh': 'NOx lbs/MWh',
    'so2_lbs_ef_mwh': 'SO2 lbs/MWh',
}

colorvar_lookup ={
    'primary_fuel':'Primary Fuel',
    'continent':'Continent'
}

groupvar_lookup = {
    'continent':'Continent',
    'plant_id_wri':'Plant Level',
    'country':'Country'
}

aggfunc_lookup = {
    'sum':'Sum',
    'mean':'Mean',
    'median':'Median',
}

# --- Dummy dfs for tables ---
plant_emission_table = pd.DataFrame({
                                'WRI Plant ID':['loading'],
                                'Primary Fuel':['loading'],
                                'Capacity (MW)':['loading'],
                                'Country':['loading'],
                                'Generation (GWh)':['loading'],
                                'CO2 (1m lbs)':['loading'],
                                'NOx (1k lbs)':['loading'],
                                'SO2 (1k lbs)':['loading'],
                                'CO2 (lbs/MWh)':['loading'],
                                'NOx (lbs/MWh)':['loading'],
                                'SO2 (lbs/MWh)':['loading'],
                                })

country_emission_table = pd.DataFrame({
                                'Country':['loading'],
                                'CO2 (1m lbs)':['loading'],
                                'NOx (1k lbs)':['loading'],
                                'SO2 (1k lbs)':['loading'],
                                })