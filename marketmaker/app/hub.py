import hug
#import sqlalchemy
import logging
from falcon import HTTP_404, HTTP_500
import json

#from .ec import EC
from pymongo import MongoClient
from pandas import DataFrame
from datetime import datetime
import os

#import schedule

from .marketmaker import convert_datetime

#mm = MarketMaker()

if 'MONGO_ADD' in os.environ:
    conn = MongoClient('mongodb://'+os.environ['MONGO_ADD'])
else:
    conn = MongoClient('mongodb://localhost:27018')


## databases
db = conn.database
ec_db = db.ec
co2_db = db.co2
svalin_db = db.svalin

def get_last_timestep():
    """Find the last time step in the MongoDB"""
    d = DataFrame(list(svalin_db.find({}, {'time'})))
    d.index = d.time
    last_time = convert_datetime(d.time.values[-1])
    return last_time

def get_past_svalin(timestep):
    """Get a specific svalin dictionary from the mongodb"""
    dic = dict(list(svalin_db.find({'time': timestep}))[0])
    return dic

def get_past_ec(timestep):
    """Get the EC dataframe from a specific timestep"""
    df = DataFrame(list(ec_db.find({'time': timestep})))
    df.index = df.ec
    return df

def get_last():
    last_time = get_last_timestep()
    df_svalin = get_past_svalin(last_time)
    return dict(df_svalin)

@hug.get('/hello')
def hello():
    logging.error("Hello called")
    return "Hello World"

@hug.get('/local_co2')
def local_co2():
    dic = get_last()
    return dic['local_co2']

@hug.get()
def getvar(var: hug.types.text):
    dic = get_last()
    if var in dic:
        return str(dic[var])
    else:
        return 'Doesnt exist'

# @hug.get('/ec/{ec_id}')
# def ec_req(request, response, ec_id: int):
#     if ec_id < 7:
#         logging.error("EC request {ec_id}".format(**locals()))
#         dic = sanitize(ec.get_data(ec_id))
#         out = json.dumps(dic)
#         col.insert_one(dic)
#         #logging.error("1.8.0:{1.8.0}".format(**out))
#         #logging.error("2.8.0:{2.8.0}".format(**out))
#         return out
#     else:
#         logging.error("No EC by id {}".format(ec_id))
#         response.status = HTTP_404
#         return HTTP_404
