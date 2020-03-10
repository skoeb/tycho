import os
import logging

import config
import fetcher
import helper
import loader
import merger
import splitter

# --- Get parent directory path ---
PDIR = os.path.join(os.getcwd(), os.pardir)

# --- Initialize Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs","sklearn_spoilage_log.txt")),
        logging.StreamHandler()
    ])

log = logging.getLogger("__init__")
log.info("starting log from __init__...")
