
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
    dcc.Store(id='selected_switches', storage_type='session'),
    dcc.Store(id='filtered_df', storage_type='session'),
    dcc.Store(id='var_df', storage_type='session'),
    dcc.Store(id='plot_dt_df', storage_type='session'),
    dcc.Store(id='plot_no_dt_df', storage_type='session'),
    dcc.Store(id='plot_map_df', storage_type='session'),
    dcc.Store(id='cf_df', storage_type='session'),
    dcc.Store(id='emission_table_df', storage_type='session'),

    # ==== Title Bar ===
    html.Div([
        
        html.Div([
            
            # --- Title ---
            html.H1(
                'Tycho Viewer',
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

        # --- Switches ---
        html.Div([

            dcc.Tabs(id='selectors', value='basic', parent_className='custom-tabs', className='custom-tabs-container',
                children=[

                    dcc.Tab(label='Basic Settings', className='custom-tab', selected_className='custom-tab--selected', value='basic',
                        children=[

                            # --- Select Variable ---
                            html.Div([
                                html.P('', style={'margin-top': 20}), #spacer
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
                                    value='nox_lbs'
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
                            ],
                            style={'margin-bottom':95},
                            ), 
                        ] # end of children
                    ), # end of tab

                    dcc.Tab(label='Advanced Settings', className='custom-tab', selected_className='custom-tab--selected', value='advanced',
                        children=[

                            # --- Select Color Column ---
                            html.Div([
                                html.P('', style={'margin-top': 20}), #spacer
                                html.P('Select variable to color powerplants:', style={'display':'inline-block'}),

                                html.Div([
                                    ' \u003f\u20dd',
                                    html.Span("For plotting purposes, how is color chosen for multiple powerplants with shared attributes.",
                                    className="tooltiptext")], className="tooltip", style={'padding-left':5}
                                ),

                                dcc.Dropdown(
                                    id='selected_colorvar',
                                    options=[
                                        {'label': 'Fuel Type', 'value':'primary_fuel'},
                                        {'label': 'Continent', 'value':'continent'},
                                        # {'label': 'Vintage Decade', 'value':'vintage'},
                                    ],
                                    value='primary_fuel'
                                )
                            ]), 

                            # --- Select Group Column ---
                            html.Div([
                                html.P('', style={'margin-top': 20}), #spacer
                                html.P('Select variable to group data in plots by:', style={'display':'inline-block'}),

                                html.Div([
                                    ' \u003f\u20dd',
                                    html.Span("For plotting purposes, how are multiple plants grouped together.",
                                    className="tooltiptext")], className="tooltip", style={'padding-left':5}
                                ),

                                dcc.Dropdown(
                                    id='selected_groupvar',
                                    options=[
                                        {'label': 'Plant', 'value':'plant_id_wri'},
                                        {'label': 'Country', 'value':'country'},
                                        # {'label': 'Vintage Decade', 'value':'vintage'},
                                    ],
                                    value='country'
                                )
                            ]), 

                            # --- Filter outliers ---
                            html.Div([
                                html.P('', style={'margin-top': 20}), #spacer
                                html.P('Hide outliers X times over mean:', style={'display':'inline-block'}),

                                html.Div([
                                    ' \u003f\u20dd',
                                    html.Span("Some outliers can skew visualizations, while these outliers can be significant and often warrant investigation into either modelling performance or operating conditions, this slider allows you to remove outliers from the plots.",
                                    className="tooltiptext")], className="tooltip", style={'padding-left':5}
                                ),

                                dcc.Slider(
                                    id='selected_outlierthresh',
                                    min=3,
                                    max=30,
                                    value=20,
                                    step=1,
                                    marks={
                                        3:{'label':'2', 'style': {'color': '#77b0b1'}},
                                        10:{'label':'10', 'style': {'color': '#77b0b1'}},
                                        20:{'label':'20', 'style': {'color': '#77b0b1'}},
                                        30:{'label':'30', 'style': {'color': '#77b0b1'}},
                                        },
                                    tooltip={'placement':'topRight'})
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
                            className='row',
                            style={'margin-bottom':20},
                            ),
                        ], #end of children
                    ), #end of tab
                ], #end of tab children
            ), #end of tabs

            # --- Button ---
            html.Button('Update Charts', id='button')
        ],
        className='six columns',
        style={'margin-bottom':100},
        ), # end of switches
    
    ],
    className='row'
    ), #end of overview and data input

    # === Map ===
    html.Div([
        
        html.Div([
            dcc.Graph(id="bubble_map")
        ],
        className='nine columns'
        ),

        html.Div([
            html.P("Highest Emission Countries:", style={'display':'inline-block'}),

            dash_table.DataTable(
                id='country_emission_table',
                columns=[{'name':i, 'id':i} for i in resources.country_emission_table.columns],
                data=resources.country_emission_table.to_dict('records'),
                # export_format = 'csv',
                style_as_list_view=True,
                style_cell={'font-family': 'Helvetica', 'font-size':'90%', 'textAlign':'center', 'maxWidth':120,'whiteSpace':'normal'},
                style_data_conditional=[
                        {
                        'if': {'row_index':'odd'},
                        'backgroundColor':'rgb(248, 248, 248)',
                        }
                    ],
                style_header={
                    'backgroundColor':'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                    },
                page_size=10,
                sort_action='native'
            ),
        ],
        className='three columns'
        ),

    ],
    className='row'
    ),


    # === Line Graph ===
    html.Div([
        dcc.Graph(id="line_graph")
    ],
    className= 'twelve columns'
    ),

    # === Scatterplot / Hive Graph ===
    html.Div([
        
        html.Div([
            dcc.Graph(id="scatter_graph")
        ],
        className='six columns'
        ),

        html.Div([
            dcc.Graph(id="violin_graph")
        ],
        className='six columns'
        ),

    ],
    className= 'twelve columns'
    ),

    # === Tables ===
    html.Div([

        html.Div([
            html.P("Highest Emission Generators:", style={'display':'inline-block'}),

            dash_table.DataTable(
                id='plant_emission_table',
                columns=[{'name':i, 'id':i} for i in resources.plant_emission_table.columns],
                data=resources.plant_emission_table.to_dict('records'),
                # export_format = 'csv',
                style_as_list_view=True,
                style_cell={'font-family': 'Helvetica', 'font-size':'90%', 'textAlign':'center', 'maxWidth':120,'whiteSpace':'normal'},
                style_data_conditional=[
                        {
                        'if': {'row_index':'odd'},
                        'backgroundColor':'rgb(248, 248, 248)',
                        }
                    ],
                style_header={
                    'backgroundColor':'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                    },
                page_size=10,
                sort_action='native'
            ),
        ],
        className='twelve columns offset by two'
        ),

    ])


]) #end of html