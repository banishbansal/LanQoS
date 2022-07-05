
#Standard Library modules
import logging
import sys
import json
import time
import platform
import tempfile
import shutil
import argparse

#Third Party modules
from pathlib import Path

input_file = { }
output_file = {
    "log": "aion.log",
    "metaData": "modelMetaData.json",
    "model": "model.pkl",
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

                    
log = None                    
def set_logger(log_file, mode='a'):                    
    global log                    
    logging.basicConfig(filename=log_file, filemode=mode, format='%(asctime)s %(name)s- %(message)s', level=logging.INFO, datefmt='%d-%b-%y %H:%M:%S')                    
    log = logging.getLogger(Path(__file__).parent.name)                    
    return log                    
                    
def get_logger():                    
    return log

        
def validateConfig(base_config):        
    config_file = Path(__file__).parent/'config.json'        
    if not Path(config_file).exists():        
        raise ValueError(f'Config file is missing: {config_file}')        
    config = read_json(config_file)        
        
    if not base_config['outputPath']:        
        base_config['outputPath'] = Path(config['outputPath'])/'production'        
        
    if 'XGBClassifier_MLBased' not in  base_config['models_info']:        
        info = {'model_name':'XGBClassifier_MLBased'}        
        info['model_path'] = Path(config['inputPath_XGBClassifier_MLBased'])        
        info['model_file'] = info['model_path']/(info['model_name'] + '_model.pkl')        
        info['performance'] = info['model_path']/(info['model_name'] + '_performance.json')        
        info['meta_data'] = info['model_path']/(info['model_name'] + '_modelMetaData.json')        
        base_config['models_info']['XGBClassifier_MLBased'] = info
    return base_config
        
class __register():        
        
    def __init__(self, input_path, output_path, model_name, meta_data, scheduler=True):        
        self.input_path = input_path        
        self.output_path = output_path        
        self.model_name = model_name        
        self.meta_data = meta_data        
        self.scheduler = scheduler        
        self.logger = get_logger()        
        self.logger.info('Running Local Registration')        
        
    def setup_registration(self):        
        pass        
        
    def get_unprocessed_runs(self, models_info):        
        scores = {}        
        self.logger.info('Unprocessed runs:')        
        for model_info in models_info.values():        
            if (model_info['performance']).exists():        
                with open(model_info['performance'], 'r') as f:        
                    data = json.load(f)        
                scores[model_info['model_name']] = data['metrices']['test_score']
                self.logger.info(f"	{model_info['model_name']} score: {data['metrices']['test_score']}")
        return scores        
    def get_best_run(self, runs_with_score):        
        best =  max(runs_with_score, key=runs_with_score.get)        
        self.logger.info(f"Best model {best} score: {runs_with_score[best]}")        
        return {'model': best, 'score' : runs_with_score[best]}        
        
    def __copy_to_output(self, source_loc, target_name=None):        
        source = Path(source_loc)        
        if source.is_file():        
            if target_name:        
                target = self.output_path/target_name        
            else:        
                target = self.output_path/(source.name)        
            shutil.copy(source, target)        
            self.logger.info(f'	copying file {source.name} to {target}')        
        
    def __register_model(self, model_info):        
        self.logger.info('Registering Model')        
        self.output_path.mkdir(parents=True, exist_ok=True)        
        meta_data = read_json(model_info['meta_data'])        
        if 'prod_files' in meta_data:        
            for file in meta_data['prod_files']:        
                if model_info['model_name'] in file:        
                    self.__copy_to_output(model_info['model_path']/file, file.split('_')[-1])        
                else:        
                    self.__copy_to_output(model_info['model_path']/file)        
        write_json(meta_data, self.output_path/output_file['metaData'])        
        
    def is_model_registered(self):        
        return (self.output_path/'model.pkl').exists()        

    def __get_registered_model_score(self):        
        data = read_json(self.output_path/'performance.json')        
        self.logger.info(f"Registered Model score: {data['metrices']['test_score']}")        
        return data['metrices']['test_score']
        
    def __force_register(self,models_info, run):        
        self.__register_model(models_info[run['model']])        
        
    def __scheduler_register(self, models_info, best_run):        
        if self.is_model_registered():        
            registered_model_score = self.__get_registered_model_score()        
            if registered_model_score >= best_run['score']:        
                self.logger.info('Registered model has better or equal accuracy')        
                return False        
        self.__register_model(models_info[best_run['model']])        
        return True        
        
    def register_model(self,models_info, best_run):        
        if self.scheduler:        
            self.__scheduler_register(models_info,best_run)        
        else:        
            self.__force_register(models_info,best_run)        
        
    def update_unprocessed(self):        
        return None
def register_model(input_path,models_info,output_path, model_name, scheduler, meta_data, ml=False):        
    if ml:        
        raise ValueError('MLFlow is wiil be supported in next release')        
        register = __ml_register(input_path, model_name, meta_data, scheduler)        
    else:        
        register = __register(input_path, output_path, model_name, meta_data, scheduler)        
    register.setup_registration()        
        
    runs_with_score = register.get_unprocessed_runs(models_info)        
    best_run = register.get_best_run(runs_with_score)        
    register.register_model(models_info, best_run)        
        
def register(base_config):        
    base_config = validateConfig(base_config)        
    key = list(base_config['models_info'].keys())[0]        
    meta_data_file = Path(base_config['models_info'][key]['meta_data'])        
    if meta_data_file.exists():        
        meta_data = read_json(meta_data_file)        
    else:        
        raise ValueError(f'Configuration file not found: {meta_data_file}')        
    usecase = meta_data['usecase']        
    output_path = Path(base_config['outputPath'])        
    output_path.mkdir(parents=True, exist_ok=True)        
    # enable logging        
    log_file = output_path/output_file['log']        
    set_logger(log_file, 'w')        
    scheduler = base_config.get('scheduler',True)        
    run_id = register_model(base_config['inputPath'],base_config['models_info'],output_path, usecase, scheduler, meta_data, ml=config['ml'])        
    status = {'Status':'Success','Message':'Model Registered'}        
    get_logger().info(f'output: {status}')        
    return json.dumps(status)
                
if __name__ == '__main__':        
    parser = argparse.ArgumentParser()        
    parser.add_argument('-o', '--outputPath', help='path for saving the output data')
    parser.add_argument('-m1','--inputPath_XGBClassifier_MLBased', action='store',help='XGBClassifier_MLBased Model Path')
    args = parser.parse_args()        
        
    config = {'inputPath':None, 'outputPath':None, 'models_info':{}, 'ml':False}        
        
    if args.outputPath:        
        config['outputPath'] = Path(args.outputPath )/'production'        
    if args.inputPath_XGBClassifier_MLBased:        
        info = {'model_name':'XGBClassifier_MLBased'}        
        info['model_path'] = Path(args.inputPath_XGBClassifier_MLBased)        
        info['model_file'] = info['model_path']/(info['model_name'] + '_model.pkl')        
        info['performance'] = info['model_path']/(info['model_name'] + '_performance.json')        
        info['meta_data'] = info['model_path']/(info['model_name'] + '_modelMetaData.json')        
        config['models_info']['XGBClassifier_MLBased'] = info
        
    try:        
        print(register(config))        
    except Exception as e:        
        if get_logger():        
            get_logger().error(e, exc_info=True)        
        status = {'Status':'Failure','Message':str(e)}        
        print(json.dumps(status))