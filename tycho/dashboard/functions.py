
# --- Python Batteries ---
import os
import json

# --- External libraries ---
import dash
import dash_table
import dash_daq as daq
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input, State
import plotly.graph_objs as go
import pandas as pd
import plotly.tools as tls
import plotly.io as pio
import json as json_func
import plotly.express as px
import pycountry_convert as pcc
import flag

# --- Module Imports ---
import tycho.dashboard.resources as resources
import tycho.dashboard.layout as layout

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~ Set up server ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# --- Theming ---
pio.templates.default = 'seaborn'

# --- Initialize App ---
app = dash.Dash(__name__)

# --- Set Name and Layout ---
app.title = 'Tycho Emissions Viewer'
app.layout = layout.html_obj

def add_flag(country):
    alpha2 = pcc.country_name_to_country_alpha2(country)
    emoji = flag.flag(alpha2)
    out = emoji + ' ' + country
    return out

# --- set up mapbox token ---
px.set_mapbox_access_token(open(os.path.join("tycho","dashboard","assets",".mapbox_token")).read())

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~ Textual Callbacks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@app.callback(
   Output('selected_methodology_text', 'children'),
   [Input('selected_methodology_switch', 'value')])
def update_type_switch_text(value):
    """Update the text of the endogenous/exogenous switch."""
    if value:
        return 'Exogenous'
    elif value == False:
        return 'Endogenous'

@app.callback(
   Output('selected_source_text', 'children'),
   [Input('selected_source_switch', 'value')])
def update_source_switch_text(value):
    """Update the text of the endogenous/exogenous switch."""
    if value:
        return 'EPA CEMS Ground Truth'
    elif value == False:
        return 'Tycho Prediction'

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~ Data Callbacks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@app.callback(
    Output('selected_switches', 'data'),
    [Input('button', 'n_clicks')],
    state=[
        State('selected_variable', 'value'),
        State('selected_fuels', 'value'),
        State('selected_daterange', 'start_date'),
        State('selected_daterange', 'end_date'),
        State('selected_aggfunc', 'value'),
        State('selected_colorvar', 'value'),
        State('selected_groupvar', 'value'),
        State('selected_methodology_text', 'children'),
        State('selected_source_text', 'children'),
        State('selected_outlierthresh', 'value')
    ])
def package_selections(n_clicks,
                       variable, fuels, startdate, enddate, aggfunc,
                       colorvar, groupvar, methodology, source, outlierthresh):
    dict_out = {'variable':variable, 
                'fuels':fuels,
                'startdate':startdate,
                'enddate':enddate,
                'aggfunc':aggfunc,
                'colorvar':colorvar,
                'groupvar':groupvar,
                'methodology':methodology,
                'source':source,
                'outlierthresh':outlierthresh,
                'variable_label':resources.variable_lookup[variable],
                'colorvar_label':resources.colorvar_lookup[colorvar],
                'groupvar_label':resources.groupvar_lookup[groupvar],
                'aggfunc_label':resources.aggfunc_lookup[aggfunc]
                }
    
    # --- On initial load, children text hasn't been updated yet ---
    if isinstance(dict_out['source'], type(None)):
        dict_out['source'] = 'Tycho Prediction'
    if isinstance(dict_out['methodology'], type(None)):
        dict_out['methodology'] = 'Endogenous'
    
    j = json.dumps(dict_out)
    return j

@app.callback(
   Output('filtered_df', 'data'),
   [Input('selected_switches', 'data')])
def filter_long_df(switches):
    """Filter a df to user selections, minus variable subset."""

    switches = json.loads(switches)

    df = resources.long_df.copy()

    # --- filter fuels ---
    df = df.loc[df['primary_fuel'].isin(switches['fuels'])]

    # --- filter dt ---
    df = df.loc[df['datetime_utc'] >= switches['startdate']]
    df = df.loc[df['datetime_utc'] <= switches['enddate']]

    # --- filter type ---
    df = df.loc[df['type'] == switches['methodology']]

    # --- filter source ---
    df = df.loc[df['source'] == switches['source']]

    # --- filter out extreme outliers ---
    df = df.loc[df['value'] < (df['value'].mean() * switches['outlierthresh'])]
    
    # --- convert datetime to str ---
    df.sort_values('datetime_utc', inplace=True)
    df['datetime_utc'] = df['datetime_utc'].astype('str')

    return df.to_json()

@app.callback(
   Output('var_df', 'data'),
   [Input('filtered_df', 'data'),
    Input('selected_switches', 'data')])
def filter_var_df(df, switches):
    """Take filtered_df and subset to variable."""

    # --- decode vars ---
    switches = json.loads(switches)
    df = pd.read_json(df)

    # --- filter variable ---
    df = df.loc[df['variable'] == switches['variable']]

    return df.to_json()

@app.callback(
   Output('plot_dt_df', 'data'),
   [Input('var_df', 'data'),
   Input('selected_switches', 'data')])
def filter_plot_dt_df(df, switches):
    """Take filtered df and group by groupvar and aggfunc with datetime_utc as observations."""
    
    # --- decode vars ---
    switches = json.loads(switches)
    df = pd.read_json(df)

    # --- groupby var ---
    aggvars = ['estimated_generation_gwh', 'wri_capacity_mw', 'value']
    groupvars = list(set(['datetime_utc','country','continent','primary_fuel'] + [switches['groupvar']]))
    df = df.groupby(groupvars, as_index=False)[aggvars].agg(switches['aggfunc'])

    return df.to_json()

@app.callback(
   Output('plot_no_dt_df', 'data'),
   [Input('var_df', 'data'),
   Input('selected_switches', 'data')])
def filter_plot_no_dt_df(df, switches):
    """Take filtered df and group by groupvar and aggfunc across all observations."""

    # --- decode vars ---
    switches = json.loads(switches)
    df = pd.read_json(df)

    # --- groupby var ---
    aggvars = ['estimated_generation_gwh', 'wri_capacity_mw', 'value']
    groupvars = list(set(['country','continent','primary_fuel','latitude', 'longitude'] + [switches['groupvar']]))
    df = df.groupby(groupvars, as_index=False)[aggvars].agg(switches['aggfunc'])

    return df.to_json()

@app.callback(
   Output('plot_map_df', 'data'),
   [Input('var_df', 'data'),
   Input('selected_switches', 'data')])
def filter_plot_map_df(df, switches):
    """Take filtered df and group by groupvar and aggfunc across all observations."""
    
    # --- decode vars ---
    switches = json.loads(switches)
    df = pd.read_json(df)

    # --- groupby var ---
    aggvars = ['estimated_generation_gwh', 'value']
    groupvars = ['plant_id_wri','wri_capacity_mw','country','continent','primary_fuel','latitude', 'longitude']
    df = df.groupby(groupvars, as_index=False)[aggvars].agg(switches['aggfunc'])

    return df.to_json()

@app.callback(
   Output('cf_df', 'data'),
   [Input('filtered_df', 'data'),
   Input('selected_switches', 'data')])
def capacity_factor_df(df, switches):
    """Calculate capacity factor for groupvar."""

    # --- decode vars ---
    switches = json.loads(switches)
    df = pd.read_json(df)

    # --- calc time diff ---
    time_diff = abs((pd.Timestamp(switches['enddate']) - pd.Timestamp(switches['startdate'])).days) + 1

    # --- subset to gross_load_mw ---
    df = df.loc[df['variable'] == 'gross_load_mw']

    # --- calc cf ---
    df['capacity_factor'] = df['value'] / (df['wri_capacity_mw'] * time_diff * 24) 
    df = df[['plant_id_wri','primary_fuel','country','continent','capacity_factor']]
    return df.to_json()

@app.callback(
   Output('country_emission_table', 'data'),
   [Input('filtered_df', 'data'),
   Input('selected_switches', 'data')])
def filter_country_emission_table_df(df, switches):
    """Assemble emissions table."""

    # --- decode vars ---
    switches = json.loads(switches)
    df = pd.read_json(df)

    # --- long to wide ---
    pivot = df.pivot_table(index='country', columns='variable', values='value').reset_index()

    # --- round ---
    pivot = pivot.round(1)

    # --- convert ---
    pivot[['nox_lbs','so2_lbs']] = pivot[['nox_lbs','so2_lbs']] / 1000
    pivot['co2_lbs'] = pivot['co2_lbs'] / 1000000

    # --- sort by var ---
    pivot.sort_values(switches['variable'], inplace=True, ascending=False)

    # --- add flags ---
    pivot['country'] = pivot['country'].apply(add_flag)

    # --- rename ---
    rename_dict = {
    'co2_lbs': 'CO2 (1m lbs)',
    'nox_lbs': 'NOx (1k lbs)',
    'so2_lbs': 'SO2 (1k lbs)',
    'country':'Country'
    } # calc cf

    pivot.rename(rename_dict, axis='columns', inplace=True)
    pivot = pivot.round(1)

    return pivot.to_dict('records')

@app.callback(
   Output('plant_emission_table', 'data'),
   [Input('filtered_df', 'data'),
   Input('selected_switches', 'data')])
def filter_plant_emission_table_df(df, switches):
    """Assemble emissions table."""

    # --- decode vars ---
    switches = json.loads(switches)
    df = pd.read_json(df)

    # --- long to wide ---
    ix_cols = ['plant_id_wri', 'primary_fuel', 'wri_capacity_mw', 'country']
    pivot = df.pivot_table(index=ix_cols, columns='variable', values='value').reset_index()

    # --- round ---
    pivot = pivot.round(1)

    # --- convert ---
    pivot[['nox_lbs','so2_lbs']] = pivot[['nox_lbs','so2_lbs']] / 1000
    pivot['co2_lbs'] = pivot['co2_lbs'] / 1000000
    pivot['gross_load_mw'] = pivot['gross_load_mw'] / 1000

    # --- sort by var ---
    pivot.sort_values(switches['variable'], inplace=True, ascending=False)

    # --- rename ---
    rename_dict = {
    'co2_lbs': 'CO2 (1m lbs)',
    'nox_lbs': 'NOx (1k lbs)',
    'so2_lbs': 'SO2 (1k lbs)',
    'co2_lbs_ef_mwh': 'CO2 (lbs/MWh)',
    'nox_lbs_ef_mwh': 'NOx (lbs/MWh)',
    'so2_lbs_ef_mwh': 'SO2 (lbs/MWh)',
    'gross_load_mw': 'Generation (GWh)',
    'plant_id_wri':'WRI Plant ID',
    'primary_fuel':'Primary Fuel',
    'wri_capacity_mw':'Capacity (MW)',
    'country':'Country'
    } # calc cf

    pivot.rename(rename_dict, axis='columns', inplace=True)
    pivot = pivot.round(1)

    return pivot.to_dict('records')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~ Plots ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@app.callback(
    Output('bubble_map', 'figure'),
   [Input('plot_map_df', 'data'),
   Input('selected_switches', 'data')])
def plot_map(df, switches):

    # --- decode vars ---
    switches = json.loads(switches)
    df = pd.read_json(df)

    fig = px.scatter_mapbox(df, lat='latitude', lon='longitude', color=switches['colorvar'],
                        size='value', hover_name='country', mapbox_style='stamen-toner',
                        hover_data=['plant_id_wri','value','wri_capacity_mw'],
                        title=f"World Map of Power Plants by {switches['aggfunc_label']} of {switches['variable_label']}",
                        height=500, zoom=1)

    # --- automatically set bounds ---
    fig.update_geos(showcountries=True, countrycolor="Black", countrywidth=0.5,
                    resolution=110) #fitbounds="locations", 

    return fig

@app.callback(
    Output('line_graph', 'figure'),
   [Input('plot_dt_df', 'data'),
   Input('selected_switches', 'data')])
def plot_line_graph(df, switches):

    # --- decode vars ---
    switches = json.loads(switches)
    df = pd.read_json(df)

    # -- set datetime ---
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])

    fig = px.line(df, x="datetime_utc", y="value", render_mode='svg',
                color=switches['colorvar'], hover_name=switches['groupvar'], line_group=switches['groupvar'],
                hover_data=list(df.columns),
                title=f"{switches['aggfunc_label']} of {switches['groupvar_label']} {switches['variable_label']}")
    
    fig.update_traces(mode='lines+markers', line_shape='spline', opacity=0.35)
    fig.update_layout(xaxis_title='Date', yaxis_title=switches['variable_label'])
    return fig

@app.callback(
    Output('scatter_graph', 'figure'),
   [Input('plot_no_dt_df', 'data'),
   Input('selected_switches', 'data')])
def scatter_plot_graph(df, switches):

    # --- decode vars ---
    switches = json.loads(switches)
    df = pd.read_json(df)

    fig = px.scatter(df, x='wri_capacity_mw', y='value',
                     color=switches['colorvar'], hover_name=switches['groupvar'], size='value', opacity=0.5,
                     marginal_x='histogram', marginal_y='histogram',
                     hover_data=list(df.columns),
                     title=f"{switches['aggfunc_label']} of {switches['groupvar_label']} Capacity<br>by {switches['aggfunc_label']} of {switches['variable_label']}")

    fig.update_layout(xaxis_title='Capacity (MW)', yaxis_title=switches['variable_label'])
    return fig

@app.callback(
    Output('violin_graph', 'figure'),
   [Input('cf_df', 'data'),
   Input('selected_switches', 'data')])
def cf_violin_graph(df, switches):

    # --- decode vars ---
    switches = json.loads(switches)
    df = pd.read_json(df)

    fig = px.violin(df, y='capacity_factor', x=switches['colorvar'], color=switches['colorvar'], box=True)
    fig.update_layout(xaxis_title=switches['colorvar_label'], yaxis_title="Capacity Factor",
                      title=f"Distribution of Capacity Factor<br>by {switches['colorvar_label']}")
    return fig

# Scatter Plot w/ variable and wri_capacity_mw, w/ marginals

# Hiveplot w/ capacity factor

# Table of top emitters

# Table of top emissions by country

# Map

# Something to show model performance on train/test set