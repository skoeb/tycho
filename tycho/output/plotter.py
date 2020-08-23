
# --- Python Batteries Included---
<<<<<<< HEAD
=======
import sqlite3
>>>>>>> 0f8b3b2bdb5120308a9454144010407128c5df28
import os
import ftplib
import concurrent.futures as cf
import time
import json
import itertools
import random
import pickle
import re

# --- External Libraries ---
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import shapely
import shap
import sklearn
from matplotlib import rcParams

# --- Module Imports ---
import tycho.config as config
import tycho.helper as helper
import tycho
# import tycho.ml.sanitizer as sanitizer
# import tycho.ml.featureengineer as featureengineer


import logging
log = logging.getLogger("tycho") 


def plot_cems_emissions(df):

    # --- run Capacity Features script ---
    cf = tycho.CapacityFeatures()
    df = cf.fit_transform(df)

    # --- Filter out a few outliers ---
    df = df.loc[df['capacity_factor'] < 1]

    # --- Get plant_id_wri and datetime_utc out of index ---
    df.reset_index(drop=False, inplace=True)

    # --- Aggregate Columns ---
    agg_dict = {
        'so2_lbs':'sum',
        'nox_lbs':'sum',
        'co2_lbs':'sum',
        'capacity_factor':'mean'
        }

    # --- Resample to Annual ---
    df = df.groupby(['plant_id_wri','primary_fuel']).resample('A', on='datetime_utc').agg(agg_dict)
    df.reset_index(inplace=True, drop=False)

    # --- Melt df ---
    melt = pd.melt(df,
                id_vars=['plant_id_wri','primary_fuel','datetime_utc','capacity_factor'], 
                value_vars=['so2_lbs','nox_lbs','co2_lbs'], 
                var_name='CEMS'
                )

    # --- Create subset dfs ---
    so2 = melt.loc[melt['CEMS'] == 'so2_lbs']
    nox = melt.loc[melt['CEMS'] == 'nox_lbs']
    co2 = melt.loc[melt['CEMS'] == 'co2_lbs']

    # --- Plot ---
    sns.set_style('darkgrid')
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(10,3), dpi=200)
    a = sns.scatterplot(x='capacity_factor', y='value', hue='primary_fuel', alpha=0.4, data=so2, ax=ax1, legend=False)
    b = sns.scatterplot(x='capacity_factor', y='value', hue='primary_fuel', alpha=0.4, data=nox, ax=ax2, legend=False)
    c = sns.scatterplot(x='capacity_factor', y='value', hue='primary_fuel', alpha=0.4, data=co2, ax=ax3)

    # --- Dot your i's ---
    ax1.set_xlabel('Capacity Factor')
    ax2.set_xlabel('Capacity Factor')
    ax3.set_xlabel('Capacity Factor')

    ax1.set_ylabel('Annual SO2 Emissions (Lbs)')
    ax2.set_ylabel('Annual NOx Emissions (Lbs)')
    ax3.set_ylabel('Annual CO2 Emissions (Lbs)')

    # --- Cross your t's ---
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    fig.suptitle('Annual U.S. Power Plant Average Capacity Factor vs. Emissions', size=20)
    fig.subplots_adjust(top=0.8)

    # --- Output ---
    plt.savefig(os.path.join('images','cems_cf_emissions.png'))

def plot_corr_heatmap(df, cems_y_cols=config.CEMS_Y_COLS):
    
    rcParams.update({'figure.autolayout': True})

    # --- Drop non correlative columns ---
    drop_cols = [
        'plant_id_eia',
        'report_year',
        'capacity_mw',
        'summer_capacity_mw',
        'winter_capacity_mw',
        'minimum_load_mw', 
        'fuel_type_code_pudl', 
        'multiple_fuels', 
        'planned_retirement_year', 
        'plant_name_eia', 
        'city', 
        'county', 
        'latitude', 
        'longitude', 
        'state', 
        'timezone', 
        'geometry', 
        'datetime_utc',
]

    df = df.drop(drop_cols, axis='columns')
    
    # --- Sort columns ignoring numbers ---
    cols = list(df.columns)
    cols = sorted(cols, key=lambda x: re.sub('[^A-Za-z]+', '', x).lower())
    df = df[cols]

    # --- Initialize Axes ---
    sns.set(style="white")
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, dpi=200,
                                        figsize=(12,60))
    
    # --- Loop through fuels ---
    for f, ax in zip(['Coal','Gas','Oil'], [ax1, ax2, ax3]):
        
        # --- Subset df ---
        f_df = df.loc[df['primary_fuel'] == f]
        f_df.drop('primary_fuel', axis='columns')
        
        # --- Compute the correlation matrix ---
        corr = f_df.corr()
            
        # --- reshape corr ---
        corr = corr[cems_y_cols]
        corr = corr.drop(cems_y_cols, axis='rows')
        corr = corr.round(2) * 100
        
        if len(corr.dropna()) > 0:
            # --- aesthetics based on position of ax ---
            if f == 'Coal':
                yticklabels=True
                cbar=False
                cbar_ax=None
            elif f == 'Gas':
                yticklabels=False
                cbar = False
                cbar_ax=None
            elif f == 'Oil':
                yticklabels=False
                cbar = True
                # cbar_ax = fig.add_axes([0.92, 0.2, 0.03, 0.6])

            # --- Draw the heatmap ---
            hm = sns.heatmap(corr, ax=ax, 
                            cmap='coolwarm',
                            center=0,
                            linewidths=.5,
                            annot=True, annot_kws={"size": 12},
                            yticklabels=yticklabels,
                            cbar=False)
                            # cbar=cbar, cbar_ax=cbar_ax)
            
            ax.set_title(f, fontsize=14)

    # --- Output ---
    plt.savefig(os.path.join('images','corr_heatmap.png'))

def plot_eda_pair(df, cems_y_cols=config.CEMS_Y_COLS):

    # rcParams.update({'figure.autolayout': True})

    # --- Subset columns ---
    df = df[cems_y_cols + ['primary_fuel']]
    
    # --- Wrapper for binned hist ---
    def constrained_hist(x, **kwargs):
        plt.hist(x, bins=8, histtype='step', linewidth=6, alpha=0.8, **kwargs)
        
    # --- Draw pairplot ---
    g = sns.pairplot(df, hue='primary_fuel')
    g.map_diag(constrained_hist)
    
    # --- Dot your i's ---
    plt.subplots_adjust(top=0.85)
    g.fig.suptitle('PairPlot of CEMS Load and Emission Data', size=20)
    
    # --- Output ---
    # plt.tight_layout()
    plt.savefig(os.path.join('images','eda_pairplot.png'))

def plot_map_plants(df,
                    country=None,
                    bounds=None,
                    title=None,
                    tiles=5):
    
    rcParams.update({'figure.autolayout': False})

    # --- force df to gdf ---
    if isinstance(df, pd.DataFrame):
        df = gpd.GeoDataFrame(df, geometry='geometry')

    # --- Subset Country ---
    if isinstance(country, str):
        df = df.loc[df['country_long'] == country]
    else:
        country = list(df['country_long'])[0]

    # --- Sort df ---    
    df = df.drop_duplicates(subset=['plant_id_wri'])
    df = df.sample(frac=1) #shuffle so that power plant colors are less clumpy
              
    # --- Map hex colors ---
    cmap= {
           'Coal':'#172121',
           'Petcoke':'#634133',
           'Oil':'#F06449',
           'Gas':'#CB793A',
        }
    
    df['color'] = df['primary_fuel'].map(cmap)
    
    # --- Normalize Capacity ---
    new_max = 400
    new_min = 20
    df['capacity_normalized'] = (new_max - new_min)/(df['wri_capacity_mw'].max() - df['wri_capacity_mw'].min()) * (df['wri_capacity_mw'] - df['wri_capacity_mw'].max()) + new_max
    mw100 = (new_max - new_min)/(df['wri_capacity_mw'].max() - df['wri_capacity_mw'].min()) * (100 - df['wri_capacity_mw'].max()) + new_max
    mw500 = (new_max - new_min)/(df['wri_capacity_mw'].max() - df['wri_capacity_mw'].min()) * (500 - df['wri_capacity_mw'].max()) + new_max
    mw1000 =(new_max - new_min)/(df['wri_capacity_mw'].max() - df['wri_capacity_mw'].min()) * (1000 - df['wri_capacity_mw'].max()) + new_max

    # --- Initialize crs ---
    crs = ccrs.InterruptedGoodeHomolosine()
    df.crs = 'EPSG:4326'
    df = df.to_crs(crs.proj4_init)
    
    # --- Get the geometry of the country ---
    country_shape = gpd.read_file(os.path.join('data','geometry','ne_110m_admin_0_countries','ne_110m_admin_0_countries.shp'))
    country_shape = country_shape.loc[country_shape['ADMIN'] == country]

    # --- Get the geometry of the country ---
    country_geo = country_shape['geometry'].item()
    
    # --- Only take the biggest polygon (i.e. no alaska or hawaii for US) ---
    if isinstance(country_geo, shapely.geometry.multipolygon.MultiPolygon):
        areas = [p.area for p in country_geo]
        biggest_area = max(areas)
        country_geo = [p for p in country_geo if p.area == biggest_area][0]

    # --- Get bounds of largest polygon ---
    bounds = country_geo.bounds
    
    # --- Clean bounds ---
    bounds = [round(i, 1) for i in bounds]
    bounds = [bounds[0]*1.02, bounds[2]*0.98, bounds[1]*0.98, bounds[3]*1.02]
    
    # --- Initialize Figure ---
    fig = plt.figure(dpi=300)
    ax = fig.add_axes([0, 0, 1, 1], projection=crs)
    ax.set_extent(bounds, ccrs.PlateCarree())
    
    # --- Add features to basemap ---  
    # land = cfeature.NaturalEarthFeature(
    #     category='physical',
    #     name='land',
    #     scale='10m',
    #     facecolor='#EFEFDB')
    
    # urban = cfeature.NaturalEarthFeature(
    #     category='cultural',
    #     name='urban_areas',
    #     scale='50m',
    #     facecolor='#F3C265')
    
    # ax.add_feature(land)
    # ax.add_feature(urban)

    #TODO: adhoc tile size selection, find a way to get the area of bounding box and select an appropriate tile size
    
    stamen_terrain = cimgt.Stamen('terrain-background')
    ax.add_image(stamen_terrain, tiles)
    ax.add_feature(cfeature.LAKES)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.STATES, linewidth=0.3)
    ax.add_feature(cfeature.BORDERS)
    
    # --- Plot power plants ---
    df.plot(color=df['color'], edgecolor='k', linewidth=0.5,
            alpha=0.4, zorder=100,
            markersize=df['capacity_normalized'],
            ax=ax, legend=True)
    
    # --- Create legend ---
    marklist = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in list(cmap.values())]
    marklabels = [f"{i} ({len(df.loc[df['primary_fuel'] == i])})" for i in cmap.keys()]
    
    plt.legend(marklist,
               marklabels,
               framealpha=0.9,
               numpoints=1,
               loc='best',
               fontsize=9)
    
    # --- Clean up ---
    if title == None:
        ax.set_title(f'GPPD/Tycho Power Plants in {country} (n={len(df)})')
    else:
        ax.set_title(title + f' (n={len(df)})')

    # --- Output ---
    country_out = country.lower().replace(' ','_')
    plt.savefig(os.path.join('images',f"{country_out}_map.png"), bbox_inches='tight')

def plot_emission_factor(data_type='pred',
                         drop_percentile=0.98,
                         country='U.S',
                         sources=['Endogenous']):
    
    assert data_type in ['train','test','pred']
    
    # --- Read df ---
    df_path = os.path.join('processed','predictions',f'{data_type}_emission_factors.csv')
    df = pd.read_csv(df_path)
    
    # --- Drop infinite values ---
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    # --- Drop other outliers ---
    var_cols = [
        'pred_co2_lbs',
        'pred_so2_lbs',
        'pred_nox_lbs',
        'pred_gross_load_mw'
    ]
    for c in var_cols:
        df = df.loc[df[c] <= df[c].quantile(drop_percentile)]
    
    # --- Convert co2 from lbs to tons ---
    co2_cols = [c for c in df.columns if 'co2_' in c]
    if 'co2_lbs' in df.columns:
        df['co2_tons'] = df['co2_lbs'] / 2000
        df.drop('co2_lbs', axis='columns', inplace=True)
    if 'exo_co2_lbs_factor_mwh' in df.columns:
        df['exo_co2_tons_factor_mwh'] = df['exo_co2_lbs_factor_mwh'] / 2000
        df.drop('exo_co2_lbs_factor_mwh', axis='columns', inplace=True)
    if 'endo_co2_lbs_factor_mwh' in df.columns:
        df['endo_co2_tons_factor_mwh'] = df['endo_co2_lbs_factor_mwh'] / 2000
        df.drop('endo_co2_lbs_factor_mwh', axis='columns', inplace=True)
    
    # --- Melt ---
    var_cols = [c for c in df.columns if ('exo_' in c) or ('endo_' in c)]
    melted = pd.melt(df, id_vars=['primary_fuel','datetime_utc'], value_vars=var_cols)
    
    # --- Create source col ---
    source = []
    for i in melted['variable']:
        if 'endo_' in i:
            source.append('Endogenous')
        elif 'exo_' in i:
            source.append('Exogenous')
            
    melted['source'] = source
    
    # --- Rename var cols ---
    var_dict = {
        'exo_so2_lbs_factor_mwh':'SO2 lbs / MWh',
        'exo_nox_lbs_factor_mwh':'NOx lbs / MWh',
        'exo_co2_tons_factor_mwh':'CO2 tons / MWh',
        'endo_so2_lbs_factor_mwh':'SO2 lbs / MWh',
        'endo_nox_lbs_factor_mwh':'NOx lbs / MWh',
        'endo_co2_tons_factor_mwh':'CO2 tons / MWh',
    }
    
    melted['variable'] = melted['variable'].map(var_dict)
    melted = melted.dropna(subset=['variable'])
    
    # --- Subset to endo/exo ---
    melted = melted.loc[melted['source'].isin(sources)]
    
    fig, ax = plt.subplots(dpi=300, figsize=(12,8))
    
    plot = sns.lineplot('datetime_utc','value', hue='variable', style='source',
                 data=melted, ax=ax)
    
    # --- Rotate X ticks ---
    plt.setp(plot.get_xticklabels(), rotation=45)
    
    # --- Change tick frequency ---
    #show only each 30th label, another are not visible
    spacing = 4
    visible = ax.xaxis.get_ticklabels()[::spacing]
    for label in ax.xaxis.get_ticklabels():
        if label not in visible:
            label.set_visible(False)
    
    # --- Move legend ---
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    # --- Set Title ---
    ax.set_ylabel('Emission Factor (lbs/tons)/MWh')
    ax.set_xlabel('')
    ax.set_title(f'{country} Tycho Emission Factor Predictions')
    
    # --- Output ---
    country_out = country.lower().replace(' ','_')
    plt.tight_layout()
    plt.savefig(os.path.join('images',f'{country_out}_emission_factor_prediction.png'))

def plot_with_map(array, transform, db):
    fig, ax = plt.subplots(figsize=(15, 15))
    countries = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    rasterio.plot.show(array, transform=transform, ax=ax)
    countries.plot(ax=ax, facecolor='none', edgecolor='red')
    ax.set_title(db)
    
def plot_shap(merged):
    # --- Setup Shap ---
    shap.initjs()

    # --- Sanitize ---
    ColumnSanitize = tycho.ColumnSanitizer()
    clean = ColumnSanitize.sanitize(merged)

    # --- Split ---
    Splitter = tycho.FourWaySplit()
    X_train_df, X_test_df, y_train_all, y_test_all = Splitter.split(clean)

    # --- Load preprocessing pipe ---
    load_path = os.path.join('models', config.PREDICT_MODEL)

    with open(os.path.join(load_path, 'pipe.pkl'), 'rb') as handle:
        preprocess_pipe = pickle.load(handle)

    # --- get feature names ---
    feature_names = preprocess_pipe['pd']['onehot'].feature_names

    X_train = preprocess_pipe.transform(X_train_df)

    # --- load models ---
    for y_col in config.ML_Y_COLS:

        # --- subset y_train ---
        y_train = y_train_all[y_col]

        with open(os.path.join(load_path, f'model_{y_col}_{config.PREDICT_MODEL}.pkl'), 'rb') as handle:
            model = pickle.load(handle)
        
        # --- grab best model from Bayes ---
        if type(model) == tycho.ml.bayes.BayesRegressor:
            model = model.best_model

        model.fit(X_train, y_train)

        # explain the model's predictions using SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)

        # --- convert X_train back to df ---
        fig, ax = plt.subplots(figsize=(10,10))
        shap.summary_plot(shap_values, X_train, feature_names=feature_names,
                        show=False, color_bar=False)
        
    #     plt.show()
        plt.tight_layout()
        
        # --- save as image ---
        image_path = os.path.join('images', f"{y_col}_shap.png")
        plt.savefig(image_path)

def plot(sql_db='tycho_production'):
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~ Read SQL ~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
<<<<<<< HEAD
    SQL = tycho.PostgreSQLCon()
=======
    SQL = tycho.SQLiteCon(sql_db)
>>>>>>> 0f8b3b2bdb5120308a9454144010407128c5df28
    SQL.make_con()

    # --- Read in ETL ---
    merged = SQL.sql_to_pandas('etl_L3')
    gppd = SQL.sql_to_pandas('gppd_merged')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~ Plot ~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    plot_cems_emissions(merged)
    plot_corr_heatmap(merged)
    plot_eda_pair(merged)
    plot_shap(merged)

    # --- Plot training data ---
    plot_map_plants(merged,
                        country='United States of America',
                        title='U.S. Power Plants Included in Tycho Training Set')

    plot_map_plants(gppd,
                        country=config.PREDICT_COUNTRIES[0],
                        tiles=10)

    # --- Plot Prediction ---
    plot_emission_factor(data_type='test', country='U.S.')

if __name__ == "__main__":
    plot()