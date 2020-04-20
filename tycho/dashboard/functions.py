
# --- External libraries
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
app.title = 'Tycho Emissions'
app.layout = layout.html_obj


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
   Output('filtered_df', 'data'),
   [Input('selected_variable', 'value'),
   Input('selected_fuels', 'value'),
   Input('selected_daterange', 'start_date'),
   Input('selected_daterange', 'end_date'),
   Input('selected_aggfunc', 'value'),
   Input('selected_methodology_text', 'children'),
   Input('selected_source_text', 'children')])
def filter_long_df(variable, fuels, startdate, enddate, aggfunc, methodology, model_source):

    df = resources.long_df.copy()
    
    # --- filter variable ---
    df = df.loc[df['variable'] == variable]

    # --- filter fuels ---
    df = df.loc[df['primary_fuel'].isin(fuels)]

    # --- filter dt ---
    df = df.loc[df['datetime_utc'] >= startdate]
    df = df.loc[df['datetime_utc'] <= enddate]

    # --- filter type ---
    df = df.loc[df['type'] == methodology]

    # --- filter source ---
    df = df.loc[df['source'] == model_source]

    # --- groupby and agg ---
    df = df.groupby(['plant_id_wri'], as_index=False).agg(aggfunc)

    return df


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~ Plots ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# def line_plot()