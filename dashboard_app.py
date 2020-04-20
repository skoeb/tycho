
# --- External libraries ---
import dash
import pandas as pd

# --- Module Imports ---
import tycho.dashboard.resources as resources
import tycho.dashboard.functions as functions

# --- Hide SettingWithCopy Warnings --- 
pd.set_option('chained_assignment',None)

# --- Server ---
server = functions.app.server
# --- Run on Import ---
if __name__ == "__main__":
    functions.app.run_server(debug=True)


"""
TODO:
- css
- header
- switches
    - source
    - type
    - variable
    - fuel
    - date range slider
    - agg func
- tables
    - top global emitters (based on switches, w/ all columns)
    - countries by variable
- line plot over time
- Hiveplot of cf and fuel
"""