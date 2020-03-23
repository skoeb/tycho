<img src="docs/astrolabe.png" width=90 align="middle"/>

# tycho-emissions

Tycho is a power sector (scope 2) emissions measurement data collection pipeline to enable supervised machine learning modelling of every power plant in the world. 

### Tycho's ETL process includes:
* Fetching data from U.S. regulatory reporting to geocode fossil fuel power plants with hourly emission profiles for SO<sub>2</sub>, NO<sub>x</sub> and CO<sub2</sub>.
* Matching U.S. regulatory reporting with the comprehensive WRI Global Power Plant Database. 
* Querying google earth engine to fetch remote sensing (satellite) data from the European Space Agency's Sentinel satellite program with emissions measurements on a daily basis across the globe. 

### Why Tycho? 
Most present-day emission measurement techniques require emprical and trusted knowledge of plant generation data (such as hourly capacity factor) to model emissions based on near-linear emission curves. These methods are vulnerable to 'garbage in garbage out' syndrome. While there are high levels of support for international climate agreements that mandate routine reporting of emissions, multiple instances of falsified or misleading emissions claims have been made by power producers, utilities, and governments. [Citation Needed] 

The state-of-the-art ESA Sentinel-5 Satellite provides remote multispectral empirical measurement of common power sector emissions including concentrations of ozone, methane, formaldehyde, aerosol, carbon monoxide, nitrogen oxide, and sulphur dioxide, as well as cloud characteristics at a spatial resolution of 0.01 arc degrees. [https://developers.google.com/earth-engine/datasets/catalog/sentinel-5p]

While the data from Sentinel-5 has been made publically available since mid-2019, it remains difficult to process and aggregate this data. Aditionally, the rasterized measurements from Sentinel can be difficult to attribute to specific power plants, or even sources of emissions (i.e. Coal plant near a city with heavy combustion engine vehicle traffic). Tycho aims to remedy this problem, by providing a clean and robust training set linking specific observations (rows) from measured power plant data with representations of Sentinel-5 observations. A well trained model using Tycho should be able to begin predicting SO<sub>2</sub>, NO<sub>x</sub> and CO<sub2</sub> (among other) emissions at weekly (or possibly daily) granularity for every significant fossil-fuel power plant across the world. 

Advantages of Tycho include:
* **Automated** querying of google earth engine for an expert-filtered and cleaned set of coordinates representing power plants from Global Power Plant Database. 
* **Robustness** from horizontal-stacking of multiple geographic scopes (1km, 10km, 100km by default) of all Sentinel data, along with observation-timed weather data (i.e. speed and direction of wind, volume of rain in the last week) along with population density for considerations of noise in data. 
* **Feature Engineering** already done for you, with expert-selected features to inform your modelling. 
* **Testing** built in to assure you that the thousands of merges going on are passing. 
* and lastly, a clean, well-documented, object-oriented code base that is extensible to novel modelling techniques. 

>This project started as a submission to the Kaggle competition [DS4G: Environmental Insights Explorer](https://www.kaggle.com/c/ds4g-environmental-insights-explorer) hosted by Google Earth Engine. 

**Badges will go here**

## Example

A complete tycho ETL process involves:
```python

# Step 1: Download EPA CEMS (hourly U.S. power plant emission data)
CemsFetch = tycho.EPACEMSFetcher()
CemsFetch.fetch() #saves as pickles

# Step 2: Load EIA 860/923 (power plant metadata and coordinates) from PUDL SQL server
PudlLoad = tycho.PUDLLoader()
PudlLoad.load()
eightsixty = PudlLoad.eightsixty

# Step 3: load CEMS data from pickles
CemsLoad = tycho.CEMSLoader()
CemsLoad.load()
cems = CemsLoad.cems

# Step 4: Load WRI Global Power Plant Database data from csv
GppdLoad = tycho.GPPDLoader() 
GppdLoad.load()
gppd = GppdLoad.gppd

# Step 5: Merge eightsixty, gppd, cems together into a long_df
TrainingMerge = tycho.TrainingDataMerger(eightsixty, gppd, cems)
TrainingMerge.merge()
df = TrainingMerge.df

# Step 6: Query Google Earth Engine Data
for earthengine_db in ["COPERNICUS/S5P/OFFL/L3_NO2", "ECMWF/ERA5/DAILY"]:
    EeFetch = tycho.EarthEngineFetcher(earthengine_db, buffers=[1e3, 1e4, 1e5])
    EeFetch.fetch(df)

# Step 7: Merge Earth Engine Data onto df
RemoteMerge = tycho.RemoteDataMerger()
df = RemoteMerge.merge(df)
```

This pipeline is contained within the `etl.py` script. User customization is included in `tycho/config.py` to specify things like `earthengine_dbs`, `buffers`, `geographies`, and `y_columns`. 

After the ETL process is completed, feature engineering and train/test splitting is performed by `process.py`
```python

# Insert code from process.py here. 
```


## Installation


## FAQs
