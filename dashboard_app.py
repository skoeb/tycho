
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
- color selection
    - fuel type
    - add continent to long_df
    - add decade of construction to long_df
- tables
    - top global emitters (based on switches, w/ all columns)
    - countries by variable
- line plot over time
- validation plot by state
- Hiveplot of cf and fuel
- engineer wind velocity
- try knn
- query point in wind direction and opposite wind direction, extra features or subtract difference? 
"""