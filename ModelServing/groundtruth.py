
import sys
import math
import json
import pandas as pd
from influxdb import InfluxDBClient
from datetime import datetime
from pathlib import Path

def is_file_name_url(file_name):                    
    supported_urls_starts_with = ('gs://','https://','http://')                    
    return file_name.startswith(supported_urls_starts_with)

class groundtruth():
    
    def __init__(self, base_config):        
        self.input_path = Path(base_config['inputPath'])        
        self.output_path = Path(base_config['outputPath'])        
        self.output_path.mkdir(parents=True, exist_ok=True)        
        self.db_enabled = False
	
    def actual(self, data=None):
        if not data:
            data = self.dataLocation
        df = pd.DataFrame()
        if Path(data).exists():
            df=read_data(data,encoding='utf-8')
        elif is_file_name_url(data):
            df = read_data(data,encoding='utf-8')
        else:
            jsonData = json.loads(data)
            df = pd.json_normalize(jsonData)
        if len(df) == 0:
            raise ValueError('No data record found')
        self.write_to_db(df)
        status = {'Status':'Success','Message':'uploaded'}
        return json.dumps(status)
		
    def write_to_db(self, data):
        if self.db_enabled:
            db = database(self.db_config)
            db.write_data(data, {'model_ver': self.model_version[0].version})
            db.close()
        else:
            output_path = self.output_path/'actuall.csv'
            data.to_csv(output_path, mode='a', header=not output_path.exists(), index=False)
			
class databse():
    def __init__(self, config):
        self.host = config['host']
        self.port = config['port']
        self.user = config['user']
        self.password = config['password']
        self.database = config['database']
        self.measurement = config['measurement']
        self.tags = config['tags']
        self.model_ver = '1'
        self.client = self.get_client()
        
    def read_data(self)->pd.DataFrame:
        cursor = self.client.query("SELECT * FROM {}".format(self.measurement))
        points = cursor.get_points()
        my_list=list(points)
        df=pd.DataFrame(my_list)
        return df
        
    def get_client(self):
        client = InfluxDBClient(self.host,self.port,self.user,self.password)
        databases = client.get_list_database()
        databases = [x['name'] for x in databases]
        if self.database not in databases:
            client.create_database(self.database)
        return InfluxDBClient(self.host,self.port,self.user,self.password, self.database)

    def write_data(self,data):
        if isinstance(data, pd.DataFrame):
            sorted_col = data.columns.tolist()
            sorted_col.sort()
            data = data[sorted_col]
            data = data.to_dict(orient='records')
        for row in data:
            if 'time' in row.keys():
                p = '%Y-%m-%dT%H:%M:%S.%fZ'
                time_str = datetime.strptime(row['time'],p)
                del row['time']
            else:
                time_str = None
            row = {k:v for k,v in row.items() if not isinstance(v, float) or not math.isnan(v)}
            if 'model_ver' in row.keys():
                self.tags['model_ver']= row['model_ver']
                del row['model_ver']
            json_body = [{
                'measurement': self.measurement,
                'time': time_str,
                'tags': self.tags,
                'fields': row
            }]
            print(json_body)
            self.client.write_points(json_body)
            
    def close(self):
        self.client.close()


if __name__ == '__main__':
    config_path = sys.argv[1]
    operation = sys.argv[2]
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        db = databse(config['db_config'])
        if operation == 'read':
            data = db.read_data()
            output = data.to_json(orient='records')
            status = {'Status':'Success','Message':json.loads(output)}
            print('read:'+json.dumps(status))
        elif operation == 'groundtruth':
            data = pd.read_csv(sys.argv[3])
            db.write_data(data)
            status = {'Status':'Success','Message':'uploaded'}
            print('groundtruth:'+json.dumps(status))
        else:
            status = {'Status':'Failure','Message':'unsupported operation'}
            print('groundtruth:'+json.dumps(status))
    except Exception as e:
        status = {'Status':'Failure','Message':str(e)}
        if operation == 'read':
            print('read:'+json.dumps(status))    
        else:
            print('groundtruth:'+json.dumps(status))
