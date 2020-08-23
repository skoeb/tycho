
import pandas as pd
import os

import tycho

import logging
log = logging.getLogger("tycho")

def wri_table(SQL):
    log.info("creating WRI table in database.")
    # --- Load WRI database and add to sql as a table ---
    wri = pd.read_csv(os.path.join('data','wri','global_power_plant_database.csv'))
    wri = wri.rename({'gppd_idnr':'plant_id_wri'}, axis='columns')
    SQL.pandas_to_sql(wri, 'wri')

def plants_view(SQL):
    log.info("creating Plants table in database.")
    query = f"""

            CREATE TABLE {SQL.schema}.plant AS
            
            WITH grouped_predictions AS (
                SELECT 
                    p.plant_id_wri, 
                    SUM(p.pred_gross_load_mw) as cum_load,
                    SUM(p.pred_so2_lbs) as cum_so2_lbs,
                    SUM(p.pred_nox_lbs) as cum_nox_lbs,
                    SUM(p.pred_co2_lbs) as cum_co2_lbs,
                    AVG(p.pred_gross_load_mw) as avg_load,
                    AVG(p.pred_so2_lbs) as avg_so2_lbs,
                    AVG(p.pred_nox_lbs) as avg_nox_lbs,
                    AVG(p.pred_co2_lbs) as avg_co2_lbs,
                    SUM(p.pred_so2_lbs) / SUM(p.pred_gross_load_mw) as so2_lbs_per_mwh,
                    SUM(p.pred_nox_lbs) / SUM(p.pred_gross_load_mw) as nox_lbs_per_mwh,
                    SUM(p.pred_co2_lbs) / SUM(p.pred_gross_load_mw) as co2_lbs_per_mwh
                FROM {SQL.schema}.predictions as p
                GROUP BY p.plant_id_wri
            )

            SELECT 
                g.*,
                g.cum_load / (w.capacity_mw * 8760) as pred_cf,
                (w.estimated_generation_gwh / 1000) / (w.capacity_mw * 8760) as wri_cf,
                w.capacity_mw,
                w.country,
                w.country_long,
                w.name,
                w.latitude,
                w.longitude,
                w.primary_fuel,
                w.other_fuel1,
                w.other_fuel2,
                w.other_fuel3,
                w.commissioning_year,
                w.owner,
                w.source,
                w.url,
                w.geolocation_source,
                w.wepp_id,
                w.year_of_capacity_data,
                w.generation_gwh_2013,
                w.generation_gwh_2014,
                w.generation_gwh_2015,
                w.generation_gwh_2016,
                w.generation_gwh_2017,
                w.estimated_generation_gwh

            FROM grouped_predictions AS g
            
            LEFT JOIN {SQL.schema}.wri AS w
                ON w.plant_id_wri = g.plant_id_wri
            """
    SQL.drop_table('plant')
    SQL.execute(query)

def country_view(SQL):
    log.info("creating Country table in database.")
    query = f"""
            CREATE TABLE {SQL.schema}.country AS
 
            SELECT 
                p.country, 
                p.country_long,
                SUM(p.cum_load) as cum_load,
                SUM(p.cum_so2_lbs) as cum_so2_lbs,
                SUM(p.cum_nox_lbs) as cum_nox_lbs,
                SUM(p.cum_co2_lbs) as cum_co2_lbs,
                SUM(p.avg_load) as avg_load,
                SUM(p.avg_so2_lbs) as avg_so2_lbs,
                SUM(p.avg_nox_lbs) as avg_nox_lbs,
                SUM(p.avg_co2_lbs) as avg_co2_lbs,
                SUM(p.cum_so2_lbs) / SUM(p.cum_load) as so2_lbs_per_mwh,
                SUM(p.cum_nox_lbs) / SUM(p.cum_load) as nox_lbs_per_mwh,
                SUM(p.cum_co2_lbs) / SUM(p.cum_load) as co2_lbs_per_mwh,
                SUM(p.generation_gwh_2013) as generation_gwh_2013,
                SUM(p.generation_gwh_2014) as generation_gwh_2014,
                SUM(p.generation_gwh_2015) as generation_gwh_2015,
                SUM(p.generation_gwh_2016) as generation_gwh_2016,
                SUM(p.generation_gwh_2017) as generation_gwh_2017,
                SUM(p.estimated_generation_gwh) as estimated_generation_gwh,
                SUM(p.cum_load) / (SUM(p.estimated_generation_gwh) * 1000) as load_diff_pct,
                COUNT(p.plant_id_wri) as n_plants
            FROM {SQL.schema}.plant AS p
            GROUP BY p.country, p.country_long
            """
    SQL.drop_table('country')
    SQL.execute(query)

def plant_pct_of_country(SQL):
    log.info("adding pct_of_country columns to plants")
    query = f"""
            ALTER TABLE {SQL.schema}.plants
            ADD pct_country_co2, pct_country_nox, pct_country_so2
            
            UPDATE {SQL.schema}.plants
            SET pct_country_co2 = {SQL.schema}.plants.cum_co2 / {SQL.schema}.country.cum_co2


            """

def dirtiest_plants_view(SQL):
    log.info("creating dirtiest table in database.")
    query = f"""
            CREATE TABLE {SQL.schema}.dirtiest AS
            WITH dirtiest AS(
                SELECT country_long, 
                    plant_id_wri,
                    co2_rank,
                    cum_co2_lbs, cum_nox_lbs, cum_so2_lbs

                FROM (SELECT ROW_NUMBER()
                    OVER(PARTITION BY country ORDER BY cum_co2_lbs DESC) AS co2_rank, *
                    FROM {SQL.schema}.plant) n
            )

            SELECT 
                d.*,
                ROUND(CAST(d.cum_co2_lbs / c.cum_co2_lbs as numeric), 6) as pct_country_co2,
                ROUND(CAST(d.cum_so2_lbs / c.cum_so2_lbs as numeric), 6) as pct_country_so2,
                ROUND(CAST(d.cum_nox_lbs / c.cum_nox_lbs as numeric), 6) as pct_country_nox
            FROM dirtiest AS d
            LEFT JOIN {SQL.schema}.country AS c
                ON d.country_long = c.country_long
            """

    SQL.drop_table('dirtiest')
    SQL.execute(query)

def database():
    SQL = tycho.PostgreSQLCon()
    wri_table(SQL)
    plants_view(SQL)
    country_view(SQL)
    dirtiest_plants_view(SQL)


if __name__ == '__main__':
    tables()
