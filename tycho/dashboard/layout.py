
# --- Python batteries ---
import os
from datetime import datetime as dt

# --- External imports ---
import dash
import dash_table
import dash_daq as daq
import dash_core_components as dcc
import dash_html_components as html

# --- Module imports ---
import tycho.dashboard.resources as resources

html_obj = html.Div([

    # === Storage ===
    dcc.Store(id='filtered_df', storage_type='session'),

    # ==== Title Bar ===
    html.Div([
        
        html.Div([
            
            # --- Title ---
            html.H1(
                'Tycho',
                style={'font-family': 'Helvetica',
                        "margin-top": "25",
                        "margin-bottom": "0"},
            ),
            
            # --- Subtitle ---
            html.H5(
                'Using satellite data to predict powerplant emissions',
                style={'font-family': 'Helvetica',
                        'position':'left',
                        'width':'100%',
                    },
            )
        ],
        className='nine columns'
        ), # end of title
        
        # --- Icon ---
        html.Img(
            src="assets/astrolabe.png",
            className='two columns',
            style={
                'height': '90px',
                'width': 'auto',
                'float': 'right',
                'position': 'relative',
                'padding-top': 5,
                # 'padding-right': 0
            }
        ), # end of image
    ],
    className='row',
    ), # end of title section

    # --- Divider ---
    html.Div([
        html.Img(
            src='assets/divider.png',
            className='twelve columns')
    ],
    className = 'twelve columns',
    style={'margin-left':'auto','margin-right':'auto'}
    ), # end of divider
    
    # === Overview and Data Input ===
    html.Div([
        
        # --- About paragraph ---
        html.Div([
            dcc.Markdown("""

                CO2 (carbon dioxide) is a high priority emission source responseible for producing a greenhouse effect in earth's atmosphere.
                NOx (nitrogen oxides) are an air polution source which interact with oxygen to create excessive troposhperic ozone buildups along with acid rain and smog.
                SO2 (sulfur dioxide) is a toxic gas that is known to have significant effects upon human health.

                Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt
                ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco
                laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in
                voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat
                non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

                Duis aute irure dolor in reprehenderit in
                voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat
                non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
                """.replace('  ', ''),
            ),
        ],
        className='six columns'
        ), # end of description paragraph

    ]), #end of overview and data input

    # --- Switches ---
    html.Div([
        
        # --- Select Variable ---
        html.Div([
            html.P('Select emission variable to plot:', style={'display':'inline-block'}),

            html.Div([
                ' \u003f\u20dd',
                html.Span("CO2, NOx, and SO2 are three smoke-stack emission sources that Tycho predicts. Tycho predicts these variables either in absolute weight (lbs), or as an emission factor (gross weight / MWh).",
                className="tooltiptext")], className="tooltip", style={'padding-left':5}
            ),

            dcc.Dropdown(
                id='selected_variable',
                options=[
                    {'label': 'CO2 lbs', 'value':'co2_lbs'},
                    {'label': 'NOx lbs', 'value':'nox_lbs'},
                    {'label': 'SO2 lbs', 'value':'so2_lbs'},
                    {'label': 'CO2 lbs / MWh', 'value':'co2_lbs_ef_mwh'},
                    {'label': 'NOx lbs / MWh', 'value':'nox_lbs_ef_mwh'},
                    {'label': 'SO2 lbs / MWh', 'value':'so2_lbs_ef_mwh'},
                ],
                value='co2_lbs_ef_mwh'
            )
        ]), 
    
        # --- Select Fuel ---
        html.Div([
            html.P('', style={'margin-top': 20}), #spacer
            html.P('Select the primary fuel of powerplants to plot:', style={'display':'inline-block'}),

            html.Div([
                ' \u003f\u20dd',
                html.Span("Almost all power sector emissions come from Coal, Oil, and Gas burning powerplants. Coal often serves as 'baseload' power, with high utilization rates (capacity factor) throughout the day. Gas is increasingly used as baseline and peaker power. Oil is used almost entirely as a peaking power source, used intermitently to meet the highest periods of electricity demand.",
                className="tooltiptext")], className="tooltip", style={'padding-left':5}
            ),

            dcc.Dropdown(
                id='selected_fuels',
                options=[{'label':f.capitalize(), 'value':f} for f in set(resources.long_df['primary_fuel'])],
                value=['Coal','Gas','Oil','Petcoke'],
                multi=True
            )
        ]),

        # --- Select Date Range ---
        html.Div([
            html.P('', style={'margin-top': 20}), #spacer
            html.P('Select the date range to consider:', style={'display':'inline-block'}),

            html.Div([
                ' \u003f\u20dd',
                html.Span("Currently, Tycho models at a monthly granularity, begining in January 2019. The first of the month contains the cumulative modeled value for that entire months worth of emissions. Future versions of Tycho may support weekly or daily models.",
                className="tooltiptext")], className="tooltip", style={'padding-left':5}
            ),

            dcc.DatePickerRange(
                id='selected_daterange',
                min_date_allowed=resources.long_df['datetime_utc'].min(),
                max_date_allowed=resources.long_df['datetime_utc'].max(),
                start_date=dt(2019,1,1),
                end_date=dt(2019,12,31),
                display_format= 'MMM, Do YYYY'
            )
        ]),

        # --- Select Aggregation Function ---
        html.Div([
            html.P('', style={'margin-top': 20}), #spacer
            html.P('Select how multiple timeslices are aggregated:', style={'display':'inline-block'}),

            html.Div([
                ' \u003f\u20dd',
                html.Span("For plotting purposes, how are multiple timeslices reduced to a single value.",
                className="tooltiptext")], className="tooltip", style={'padding-left':5}
            ),

            dcc.Dropdown(
                id='selected_aggfunc',
                options=[
                    {'label': 'Cumulative Sum', 'value':'sum'},
                    {'label': 'Mean', 'value':'mean'},
                    {'label': 'Median', 'value':'median'},
                ],
                value='sum'
            )
        ]), 

        # --- Select Endogenous / Exogenous ---
        html.Div([
            html.P('', style={'margin-top': 20}), #spacer
            html.P('Select emission factor methodology:', style={'display':'inline-block'}),

            html.Div([
                ' \u003f\u20dd',
                html.Span("Tycho predicts an endogenous emission factor by seperately modelling the powerplants gross load (MWh) and emissions (lbs) to produce predictions that are sensitive to changes in generation. Exogenous predictions take the emissions (lbs) and divide them by the sliced annual WRI generation estimate for the given powerplant (i.e. annual estimate / 12 for monthly).",
                className="tooltiptext")], className="tooltip", style={'padding-left':5}
            ),

            html.Div([
                
                daq.ToggleSwitch(
                    id='selected_methodology_switch',
                    value=False,
                    className='three columns',
                    size=50
                ),

                html.Div(id='selected_methodology_text', className='two columns')
            ])
        ],
        className='row'
        ),

        # --- Select Source ---
        html.Div([
            html.P('', style={'margin-top': 20}), #spacer
            html.P('Select data source to plot:', style={'display':'inline-block'}),

            html.Div([
                ' \u003f\u20dd',
                html.Span("Ground Truth data from the EPA CEMS program is used to train Tycho based on U.S. only values. Tycho predictions are available for every powerplant in the WRI global powerplant database.",
                className="tooltiptext")], className="tooltip", style={'padding-left':5}
            ),

            html.Div([
                
                daq.ToggleSwitch(
                    id='selected_source_switch',
                    value=False,
                    className='three columns',
                    size=50
                ),

                html.Div(id='selected_source_text', className='five columns')
            ])
        ],
        className='row'
        ),

    ],
    className='six columns'
    )


]) #end of html