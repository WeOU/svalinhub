import requests as req
from xml.etree import ElementTree
from time import sleep
import pandas as pd
from pandas import DataFrame, read_json, concat
from pymongo import MongoClient
from collections import defaultdict
import time
import os


#%load_ext dotenv
import numpy as np
from  datetime import datetime
try:
    from dotenv import load_dotenv
    from pathlib import Path  # python3 only
    env_path = Path('..') / '.env'
    load_dotenv(dotenv_path=env_path)
except:
    pass

DEFAULT_GRID_CO2 = 400

def extract(resp):
    try:
        return ElementTree.fromstring(resp.content).find('value').text
    except Exception as e:
        raise Exception('Extract failed', str(e), str(resp))

#ip = '127.0.0.1'
ip = '10.170.111.0'
str_power_prox = 'http://{ip}:808{k}/typebased_WS_UtilityMeter/UtilityMeterWebService/Meter_EC0{k}/getACActivePower'
str_net_prox = 'http://{ip}:808{k}/typebased_WS_UtilityMeter/UtilityMeterWebService/Meter_EC0{k}/getNetActiveEnergy'
str_export_prox = 'http://{ip}:808{k}/typebased_WS_UtilityMeter/UtilityMeterWebService/Meter_EC0{k}/getActiveEnergyExport'
str_import_prox = 'http://{ip}:808{k}/typebased_WS_UtilityMeter/UtilityMeterWebService/Meter_EC0{k}/getActiveEnergyImport'
str_solar_export_prox = 'http://{ip}:808{k}/typebased_WS_PV/PVSystemWebService/PV_EC0{k}/getActiveEnergyCounter'
str_solar_power_prox = 'http://{ip}:808{k}/typebased_WS_PV/PVSystemWebService/PV_EC0{k}/getACActivePower'

#proxies = {'http': 'socks5://localhost:8088'}

def convert_datetime(npdatetime):
    ts = (npdatetime - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    return datetime.utcfromtimestamp(ts)

def get_prox(str_, k, ip):
    try:
        #return extract(req.get(str_.format(k=k, ip=ip), proxies=proxies))
        return extract(req.get(str_.format(k=k, ip=ip)))
    except Exception as e:
        if k in [3, 5] and 'PV' in str_:
            pass
        else:
            print(e, str_, k, ip)

        #get = lambda str_, k, ip: extract(req.get(str_.format(k=k, ip=ip)))

def null_if_nan(val):
    if val == np.NaN:
        return 0
    else:
        return val

def add_time(data, t):
    """Add the details of time in a dict in order to easily look them up in the mongodb
    """
    data['year'] = t.year
    data['month'] = t.month
    data['day'] = t.day
    data['hour'] = t.hour
    data['minute'] = t.minute
    data['second'] = t.second

def same_day():
    now = datetime.now()
    return {'year': now.year, 'month': now.month, 'day': now.day}

def dictify(df):
    """Catch the edge case where the data keys are somehow stored as interger
    instead of strings"""
    return {str(k): v for k, v in df.items()}

class MarketMaker(object):

    EC_translation = {
        '2': 'House2',
        '3': 'CommonHouse',
        '4': 'House12',
        '5': 'House3',
        '6': 'House4',
        #7: 'House5',
        #8: 'CarCharger'
    }

    def __init__(self, battery_capacity=0.0):
        ## Connect to the MongoDBs

        #self.debts = []
        #self.timesteps = []

        #conn = MongoClient('mongodb://%s:%s@mongodb' % ('username', 'password'))
        if 'MONGO_ADD' in os.environ:
            print('Using mongo server:', 'mongodb://'+os.environ['MONGO_ADD'])
            self.conn = MongoClient('mongodb://'+os.environ['MONGO_ADD'])
        else:
            print('Using mongo server:', 'mongodb://localhost:27018')
            self.conn = MongoClient('mongodb://localhost:27018')
        #try:

        #except:
        #    self.conn = MongoClient('mongodb://localhost:27018')

        ## databases
        self.db = self.conn.database
        self.ec_db = self.db.ec
        self.co2_db = self.db.co2
        self.svalin_db = self.db.svalin

        self.imp = {}
        self.exp = {}

        ## Restore state from last MongoDB log
        try:
            debts = list(self.svalin_db.find(same_day(), {'debt', 'time'}))
            self.debts = [read_json(d['debt']) for d in debts if 'debt' in d]
            self.timesteps = [d['time'] for d in debts if 'debt' in d]
            last_time = self.get_last_timestep()
            df_ec = self.get_past_ec(last_time)
            df_ec.index=df_ec.ec
            self.imp = dictify(df_ec.imp)
            self.exp = dictify(df_ec.exp)
            self.dimp = dictify(df_ec.dimp)
            self.dexp = dictify(df_ec.dexp)
            self.grid_co2 = df_ec.grid_co2.values[0]
            self.grid_co2_time = last_time
            print('imp: ', self.imp)
            print('exp: ', self.exp)

            self.df = self.get_df(self.dimp, self.dexp)
            self.market = EgalitarianSvalin(self.df, self.grid_co2)

        except Exception as e:
            print('# Warning: could not restore market state', str(e))
            self.dimp = defaultdict(lambda: 0.0)
            self.dexp = defaultdict(lambda: 0.0)
            self.grid_co2 = DEFAULT_GRID_CO2
            self.imp = defaultdict(lambda: 0.0)
            self.exp = defaultdict(lambda: 0.0)
            self.market = EgalitarianSvalin()
            last_time = datetime.now()
            self.grid_co2_time = last_time


        self.current_time = last_time

        # A list of previous days energy debts
        self.previous_days = []

        # The current day energy debt dictionary
        # key: (house1, house2), value: amount of debt in kWh of house1 with respect of house2
        self.energy_debt = defaultdict(lambda: 0.0)

        #self.step()

    def get_last_timestep(self):
        """Find the last time step in the MongoDB"""
        d = DataFrame(list(self.svalin_db.find({}, {'time'})))
        d.index = d.time
        last_time = convert_datetime(d.time.values[-1])
        return last_time

    def get_past_svalin(self, timestep):
        """Get a specific svalin dictionary from the mongodb"""
        dic = dict(list(self.svalin_db.find({'time': timestep}))[0])
        return df

    def get_past_ec(self, timestep):
        """Get the EC dataframe from a specific timestep"""
        df = DataFrame(list(self.ec_db.find({'time': timestep})))
        df.index = df.ec
        return df

    def new_day(self):
        """Start a new day by resetting the energy_debt to zero"""
        self.previous_days.append(self.energy_debt)
        self.energy_debt = defaultdict(lambda: 0.0)

        #TODO: add the settelement mechanism here

    def step(self):
        ## Save up the previous import/export for calculating the differentials
        self.old_exp = self.exp
        self.old_imp = self.imp
        self.old_time = self.current_time

        data_ec = {}
        self.current_time = datetime.now()
        self.timesteps.append(self.current_time)

        ## Poll the current gCO2/kWh in the grid
        if (self.current_time - self.grid_co2_time).seconds > 5.0 * 60.0:
            self.grid_co2, co2_data = self.poll_co2()
            self.grid_co2_time = co2_data['time']

        ## Prepare the data_ec dictionary
        for iEC in self.EC_translation.keys():
            data_ec[iEC] = {
                'ec': iEC,
                'time': self.current_time,
                'dt' : (self.current_time - self.old_time).seconds}
            add_time(data_ec[iEC], data_ec[iEC]['time'])

        ## poll_houses(self)
        self.imp, self.exp = self.poll_houses(data_ec)


        self.dimp, self.dexp = self.calculate_diff(self.imp, self.exp, self.old_imp, self.old_exp)

        for iEC in self.dimp.keys():
            data_ec[iEC]['dimp'] = self.dimp[iEC]
            data_ec[iEC]['dexp'] = self.dexp[iEC]

        # Create the main differential DataFrame
        self.df = self.get_df(self.dimp, self.dexp)

        ## Calculate the current market
        self.market.update(self.df, self.grid_co2)
        current_debt = self.market.get_dfdebt(self.market.current_debt)
        self.debts.append(current_debt)

        ## Compute the local CO2 level
        co2_data['localCarbonIntensity'] = self.market.local_co2

        ## Put the current CO2 level in the CO2 database
        self.co2_db.insert_one(co2_data)

        ## Add the EC data to the MongoDB
        for iEC in self.EC_translation.keys():
            data_ec[iEC]['grid_co2'] = self.grid_co2
            data_ec[iEC]['local_co2'] = self.market.local_co2
            self.ec_db.insert_one(data_ec[iEC])

        ## Log the current status for Svalin
        data_svalin = {
            'local_co2': self.market.local_co2,
            'grid_co2': self.market.grid_co2,
            'current_energy_produced': self.market.current_energy_produced,
            'current_energy_consumed': self.market.current_energy_consumed,
            'imp_grid': self.market.import_grid   ,
            'exp_grid': self.market.export_grid,
            'local_cons': self.market.local_cons,
            'ratio_local_cons': self.market.ratio_local_cons,
            'ratio_local_prod': self.market.ratio_local_prod,
            'price_per_kwh': self.market.price_per_kwh,
            'time': self.current_time,
            'dt': (self.current_time - self.old_time).seconds,
            'imp': self.imp,
            'exp': self.exp,
            'dimp': self.dimp,
            'dexp': self.dexp,
            'debt': current_debt.to_json()
        }
            #'debt': list(self.market.current_debt.items()),
        add_time(data_svalin, data_svalin['time'])
        self.svalin_db.insert_one(data_svalin)
        self.market.status()

        # For debugging purpose
        self.co2_data = co2_data
        self.svalin_data = data_svalin
        self.ec_data = data_ec

    def calculate_diff(self, imp, exp, old_imp, old_exp):
        ## Initialize the energy differentials dictionaries
        dimp = {}
        dexp = {}
        #data = {}

        for iEC in self.EC_translation.keys():
            try:
                dimp[iEC] = float(imp[iEC]) - float(old_imp[iEC])
                dexp[iEC] = float(exp[iEC]) - float(old_exp[iEC])
            except Exception as e:
                print('# WARNING: Couldnt calculate the differential of', iEC, str(e))
                dimp[iEC] = 0.
                dexp[iEC] = 0.

        return dimp, dexp

    def get_df(self, dimp, dexp):
        return pd.DataFrame([{'name': self.EC_translation[str(k)], 'imp': dimp[k], 'exp': dexp[k]} for k in dimp.keys()])

    def poll_co2(self):
        """Call Tomorrow to get the current status of the gCO2/kWh in the grid

        Returns:
        --------
        co2_data: dict['carbonIntensity', 'year'...'second']
            A data dictionary containing the output of the CO2signal response as well as some date information
        """
        response = req.get('https://api.co2signal.com/v1/latest?countryCode=DK-DK2', headers={'auth-token': os.environ['CO2_AUTH']})
        carbon_time = datetime.now()
        resp = response.json()
        try:
            data = resp['data']
            data['time'] = datetime.now()
            if 'carbonIntensity' in data:
                carbon = data['carbonIntensity']
            else:
                if hasattr(self, 'grid_co2'):
                    print('Warning: using previous step grid CO2')
                    carbon = self.grid_co2
                    data['carbonIntensity'] =  carbon
                else:
                    print('Warning: using DEFAULT_GRID_CO2')
                    carbon = DEFAULT_GRID_CO2
                    data['carbonIntensity'] =  carbon
            add_time(data, data['time'])
            #self.grid_co2 = carbon
            #co2_data = data
        except Exception as e:
            print('Error: couldnt retrieve current CO2 intensity')
            print(str(e))
            if hasattr(self, 'grid_co2'):
                print('Warning: using previous step grid CO2')
                carbon = self.grid_co2
            else:
                print('Warning: using DEFAULT_GRID_CO2')
                carbon = DEFAULT_GRID_CO2

            data = {}
            data['time'] = datetime.now()
            add_time(data, data['time'])
            data['carbonIntensity'] = carbon
            #self.grid_co2 = DEFAULT_GRID_CO2
            #co2_data = data
        return data['carbonIntensity'], data

    def poll_houses(self, data_ec):
        """Poll all the houses for their current data

        Parameters:
        -----------------------
        data_ec: dict[iEC]['imp', 'exp', 'P', 'net', 'P_solar', 'P_exp']
            A dictionary of data for each EC, will be updated during the call

        Returns:
        --------
        imp: dict[iEC]
            The total import of energy of EC iEC in kWh
        exp: dict[iEC]
            The total export of energy of EC iEC in kWh
        """
        # Add additional data
        extra_data = {
            'imp': str_import_prox,
            'exp': str_export_prox,
            'P': str_power_prox,
            'net': str_net_prox,
            'P_solar': str_solar_power_prox,
            'P_exp': str_solar_export_prox}

        for k, v in extra_data.items():
            for iEC in self.EC_translation.keys():
                data_ec[iEC][k] = get_prox(v, iEC, ip)

        imp = {}
        exp = {}
        for iEC in self.EC_translation.keys():
            if data_ec[iEC]['imp'] is None:
                print('WARNING: Using previous timestep import', iEC)
                imp[iEC] = self.old_imp[iEC]
            else:
                imp[iEC] = data_ec[iEC]['imp']
            if data_ec[iEC]['exp'] is None:
                print('WARNING: Using previous timestep export', iEC)
                exp[iEC] = self.old_exp[iEC]
            else:
                exp[iEC] = data_ec[iEC]['exp']
        return imp, exp

    def get_energy_debt(self):
        return concat(self.debts).pivot_table('amount', 'to', 'from', np.sum)

class EgalitarianSvalin(object):
    PV_CO2 = 60.0 # gCO2/kWh -- Amount of gCO2/kWh accounted for the PV production
    MAX_CO2 = 450.0 # gCO2/kWh -- Max amount of gCO2/kWh in DK

    def __init__(self, df=None, grid_co2=None):
        """
        Parameters:
        -----------
        df: DataFrame
            The current Dataframe of Svalin EC differentials
        """
        if df is not None and grid_co2 is not None:
            self.df = df
            self.grid_co2 = grid_co2
            self.compute()

    def update(self, df, grid_co2):
        self.df = df
        self.grid_co2 = grid_co2
        self.compute()

    @property
    def current_energy_produced(self):
        """Current Energy Produced in Svalin"""
        return self.df.exp.sum()

    @property
    def current_energy_consumed(self):
        """Current Energy Consumed in Svalin"""
        return self.df.imp.sum()

    @property
    def import_grid(self):
        """Energy imported from the Grid"""
        return max(0, self.current_energy_consumed - self.current_energy_produced)

    @property
    def export_grid(self):
        """Energy locally consumed"""
        return max(0, self.current_energy_produced -  self.current_energy_consumed)

    @property
    def local_cons(self):
        """Energy locally consumed"""
        return self.current_energy_produced - self.export_grid

    @property
    def ratio_local_cons(self):
        """Ratio of energy consumed which is locally produced"""
        if self.current_energy_consumed == 0.0:
            return 1.0
        else:
            return self.local_cons / self.current_energy_consumed

    @property
    def ratio_local_prod(self):
        """Ratio of energy produced consumed locally"""
        if self.current_energy_produced == 0.0:
            return 1.0
        else:
            return 1. - self.export_grid / self.current_energy_produced

    @property
    def price_per_kwh(self):
        return 0.6 + (2.2-0.6)*(self.local_co2 - self.PV_CO2)/(self.MAX_CO2 - self.PV_CO2)

    @property
    def local_co2(self):
        """Compute local CO2 intensity"""
        return self.PV_CO2 * self.ratio_local_cons + self.grid_co2 * (1. - self.ratio_local_cons)

    def status(self):
        print('- Current Energy Produced in Svalin', self.current_energy_produced, 'kWh')
        print('- Current Energy Consumed in Svalin', self.current_energy_consumed, 'kWh')
        print('- Energy imported from the Grid:', self.import_grid, 'kWh')
        print('- Energy exported to the Grid:', self.export_grid, 'kWh')
        print('- Energy locally consumed', self.local_cons, 'kWh')
        print('- Ratio of energy consumed which is locally produced', self.ratio_local_cons * 100., '%')
        print('- Ratio of energy produced which is consumed locally', self.ratio_local_prod * 100., '%')
        print('Energy Debt:')
        try:
            print(str(self.get_energy_debt()))
        except:
            pass

    def compute(self):
        """Primitive market maker assuming that all energy is first consumed
        locally and that the surplus is spread evenly amoung all the consuming
        houses of Svalin

        Class variables written:
        -----------------------
        current_debt: dict[iEC1, iEC2] iEC1 & iEC2 in [1..N, 'Grid', 'Car']
            Current debt of energy from iEC1 to iEC2
        """
        current_debt = {}
        for k in self.df.name.values:
            impk = null_if_nan(self.df[self.df.name == k].imp.values[0])
            expk = null_if_nan(self.df[self.df.name == k].exp.values[0])
            for f in self.df.name.values:
                expf = null_if_nan(self.df[self.df.name == f].exp.values[0])
                if self.df.exp.sum() > 0.0:
                    current_debt[k, f] = impk * self.ratio_local_cons * self.ratio_local_prod * expf / self.df.exp.sum()
                else:
                    current_debt[k, f] = 0.0
                #data.append({'from': k, 'to': f, 'amount': self.energy_debt[k, f]})

            current_debt[k, 'Grid'] = (1. - self.ratio_local_cons) * impk
            current_debt['Grid', k] = (1. - self.ratio_local_prod) * expk
            #     self.energy_debt[k, f] + current_debt[k, f]
            # self.energy_debt[k, 'Grid'] += current_debt[k, 'Grid']
            # self.energy_debt['Grid', k] += current_debt['Grid', k]
            #self.grid_deb[k] += (1-ratio_local_prod) * expf
            #self.debt_to_grid[k] += (1.-ratio_local_cons) * impk
            #data.append({'from': k, 'to': 'Grid', 'amount': debt_to_grid[k]})
            #data.append({'from': 'Grid', 'to': k, 'amount': grid_deb[k]})

        self.current_debt = current_debt
        #self.current_dfdebt = self.get_dfdebt(current_debt)
        return current_debt

    def get_dfdebt(self, energy_debt):
        """Create a energy Debt dataframe out of an energy_debt dictionary"""
        return pd.DataFrame([{'from': k[0], 'to': k[1], 'amount': v} for k, v in energy_debt.items()])

if __name__ == '__main__':
    mm = MarketMaker()
    while True:
        time.sleep(30)
        mm.step()
