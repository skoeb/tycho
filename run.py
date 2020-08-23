# --- Python Batteries Included---
import os
import argparse

# --- logging ---
import logging
log = logging.getLogger("tycho") 

# --- Module Imports ---
import tycho

<<<<<<< HEAD
def main(etl=False, train=False, predict=False, plot=False, dashboard=False, database=False):
=======
def main(etl=False, train=False, predict=False, plot=False, dashboard=False):
>>>>>>> 0f8b3b2bdb5120308a9454144010407128c5df28
    """Train, Predict, Merge, Calc Emissions."""

    if etl:
        log.info('====== Begining ETL ======')
        tycho.etl()
    
    if train:
        log.info('====== Begining Train ======')
        tycho.train()

    if predict:
        log.info('====== Begining Predict ======')
        tycho.predict()

    if plot:
        log.info('====== Begining Plot ======')
        tycho.plot()

    if dashboard:
        log.info('====== Begining Dashboard ======')
        tycho.package()

<<<<<<< HEAD
    if database:
        log.info('====== Begining Database ======')
        tycho.database()

=======
>>>>>>> 0f8b3b2bdb5120308a9454144010407128c5df28

if __name__ == "__main__":
    # --- CLI arguments ---
    parser = argparse.ArgumentParser(description='Full Pipeline for Tycho including ETL, Train, Predict, Plot, and host local dashboard.')
    parser.add_argument('-e','--etl', action='store_true')
    parser.add_argument('-t','--train', action='store_true')
    parser.add_argument('-pr','--predict', action='store_true')
    parser.add_argument('-pl','--plot', action='store_true')
    parser.add_argument('-d','--dashboard', action='store_true')
<<<<<<< HEAD
    parser.add_argument('-db', '--database', action='store_true')

=======
>>>>>>> 0f8b3b2bdb5120308a9454144010407128c5df28

    # --- Parse args ---
    args = parser.parse_args()
    args_dict = vars(args)
    
    # --- Run Tycho ---
    main(**args_dict)

