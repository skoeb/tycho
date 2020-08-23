import sqlite3
import sqlalchemy as sa
import pandas as pd
import shutil
import os

import tycho.config as config

import logging
log = logging.getLogger("tycho")

class SQLBase():
    def close_con(self):
        self.con.close()

    def drop_table(self, table):
        self.make_con()

        if self.schema != None:
            table = self.schema + '.' + table
            
        try:
            self.con.execute(f"DROP TABLE IF EXISTS {table}")
        except Exception as e: #OperationalError
            self.con.execute(f"DROP VIEW IF EXISTS {table}")
        self.close_con()
    
    def pandas_to_sql(self, df, table):

        self.make_con()

        _df = df.copy()

        # --- cast geometry as str ---
        if 'geometry' in df.columns:
           _df['geometry'] = _df['geometry'].astype('str')

        self.drop_table(table)

        self.make_con()
        _df.to_sql(table, self.con, if_exists='replace', index=False, schema=self.schema)

        self.close_con()

    
    def execute(self, query):
        self.make_con()
        self.con.execute(query)
        self.close_con()


    def sql_to_pandas(self, table):

        self.make_con()

        try:
            df = pd.read_sql_table(table, self.con, schema=self.schema)
        except Exception as e:
            df = pd.read_sql(f"select * from {table}", self.con)

        if 'geometry' in df.columns:
            df['geometry'] = df['geometry'].apply(wkt.loads)
            df = gpd.GeoDataFrame(df, geometry='geometry')
        
        if 'datetime_utc' in df.columns:
            df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])

        self.close_con()

        return df


class SQLiteCon(SQLBase):

    def __init__(self, db_path, db_type='.sqlite'):

        assert isinstance(db_path, str)
        if db_path.endswith(db_type):
            pass
        else:
            db_path = db_path + db_type

        self.db_path = db_path
        self.schema=None

    def make_con(self):
        self.con = sqlite3.connect(os.path.join('data','sqlite', self.db_path), timeout=10)
        return self


class PostgreSQLCon(SQLBase):

    def __init__(self, schema=config.SCHEMA):
        self.schema=schema

    def make_con(self):
        db_conn_str = os.environ.get('DB_CONN_STR') #set env variable
        engine = sa.create_engine(db_conn_str)
        self.con = engine.connect()
        return self



