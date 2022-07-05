
#Standard Library modules
import sys
import math
import json
import shutil
import platform

#Third Party modules
import joblib
import mlflow
import sklearn
import numpy as np 
import pandas as pd 
from pathlib import Path
from influxdb import InfluxDBClient
from xgboost import XGBClassifier

input_file = {
    "inputData": "rawData.dat",
    "metaData": "modelMetaData.json",
    "performance": "performance.json"
}
output_file = {
    "performance": "performance.json"
}
                    
def read_json(file_path):                    
    data = None                    
    with open(file_path,'r') as f:                    
        data = json.load(f)                    
    return data                    
                    
def write_json(data, file_path):                    
    with open(file_path,'w') as f:                    
        json.dump(data, f)                    
                    
def read_data(file_path, encoding='utf-8', sep=','):                    
    return pd.read_csv(file_path, encoding=encoding, sep=sep)                    
                    
def write_data(data, file_path, index=False):                    
    return data.to_csv(file_path, index=index)                    
                    
#Uncomment and change below code for google storage                    
#def write_data(data, file_path, index=False):                    
#    file_name= file_path.name                    
#    data.to_csv('output_data.csv')                    
#    storage_client = storage.Client()                    
#    bucket = storage_client.bucket('aion_data')                    
#    bucket.blob('prediction/'+file_name).upload_from_filename('output_data.csv', content_type='text/csv')                    
#    return data                    
                    
def is_file_name_url(file_name):                    
    supported_urls_starts_with = ('gs://','https://','http://')                    
    return file_name.startswith(supported_urls_starts_with)                    


class database():
    def __init__(self, config):
        self.host = config['host']
        self.port = config['port']
        self.user = config['user']
        self.password = config['password']
        self.database = config['database']
        self.measurement = config.get('measurement', 'measurement')
        self.tags = config['tags']
        self.client = self.get_client()

    def get_client(self):
        client = InfluxDBClient(self.host,self.port,self.user,self.password)
        databases = client.get_list_database()
        databases = [x['name'] for x in databases]
        if self.database not in databases:
            client.create_database(self.database)
        return InfluxDBClient(self.host,self.port,self.user,self.password, self.database)

    def write_data(self,data, tags={}):
        if isinstance(data, pd.DataFrame):
            data = data.to_dict(orient='records')
        for row in data:
            json_body = [{
                'measurement': self.measurement,
                'tags': dict(self.tags, **tags),
                'fields': row
            }]
            res = self.client.write_points(json_body)

    def close(self):
        self.client.close()


class deploy():

    def __init__(self, base_config):        
        self.input_path = Path(base_config['inputPath'])        
        self.output_path = Path(base_config['outputPath'])        
        self.output_path.mkdir(parents=True, exist_ok=True)        
        self.db_enabled = False        
        self.dataLocation = self.input_path/input_file['inputData']        
        meta_data_file = self.input_path/input_file['metaData']        
        if meta_data_file.exists():        
            meta_data = read_json(meta_data_file)        
        else:        
            raise ValueError(f'Configuration file not found: {meta_data_file}')        
        self.usecase = meta_data['usecase']        
        self.selected_features = meta_data['load_data']['selected_features']        
        self.train_features = meta_data['training']['features']
        self.missing_values = meta_data['transformation']['fillna']
        self.target_encoder = joblib.load(self.input_path/meta_data['transformation']['target_encoder'])
        self.model = joblib.load(self.input_path/'model.pkl')
        if not Path(self.output_path/output_file['performance']).exists():
            shutil.copy(Path(self.input_path/input_file['performance']), Path(self.output_path/output_file['performance']))

    def write_to_db(self, data):
        if self.db_enabled:
            db = database(self.db_config)
            db.write_data(data, {'model_ver': self.model_version[0].version})
            db.close()
        else:
            output_path = self.output_path/'prediction.csv'
            data.to_csv(output_path, mode='a', header=not output_path.exists(), index=False)

    def predict(self, data=None):
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
        missing_features = [x for x in self.selected_features if x not in df.columns]
        if missing_features:
            raise ValueError(f'some feature/s is/are missing: {missing_features}')
        df_copy = df.copy()
        df = df[self.selected_features]
        df.fillna(self.missing_values, inplace=True)
        df = df[self.train_features]
        df = df.astype(np.float32)		
        output = pd.DataFrame(self.model.predict_proba(df), columns=self.target_encoder.classes_)        
        df_copy['prediction'] = output.idxmax(axis=1)        
        self.write_to_db(df_copy)        
        df_copy['probability'] = output.max(axis=1).round(2)        
        df_copy['remarks'] = output.apply(lambda x: x.to_json(), axis=1)        
        output = df_copy.to_json(orient='records')
        return output        
        
if __name__ == '__main__':        
    parser = argparse.ArgumentParser()        
    parser.add_argument('-i', '--inputPath', help='path of the input data')        
    parser.add_argument('-u', '--inputUri', help='uri for the input data')        
    parser.add_argument('-o', '--outputPath', help='path for saving the output data')        
        
    args = parser.parse_args()        
        
    config = {'inputPath':None, 'outputPath':None}        
        
    if args.inputPath:        
        config['inputPath'] = Path(args.inputPath)/'production'        
    if args.inputUri: #uri has higher preference than input path        
        config['inputPath'] = args.inputUri        
    if args.outputPath:        
        config['outputPath'] = args.outputPath        
    try:        
        predictor = deploy(config)        
        output = predictor.predict()        
        status = {'Status':'Success','Message':json.loads(output)}        
        print('predictions:'+json.dumps(status))        
    except Exception as e:        
        status = {'Status':'Failure','Message':str(e)}        
        print('predictions:'+json.dumps(status))