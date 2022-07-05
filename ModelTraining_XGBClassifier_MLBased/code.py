
#Standard Library modules
import argparse
import logging
import shutil
import importlib
import operator
import platform
import time
import sys
import json
import math

#Third Party modules
import joblib
import pandas as pd 
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import mlflow
import sklearn

model_name = 'XGBClassifier_MLBased'

input_file = {
    "inputData": "featureEngineeredData.dat",
    "metaData": "modelMetaData.json"
}
output_file = {
    "log": "XGBClassifier_MLBased_aion.log",
    "metaData": "XGBClassifier_MLBased_modelMetaData.json",
    "model": "XGBClassifier_MLBased_model.pkl",
    "performance": "XGBClassifier_MLBased_performance.json"
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

                    
def add_file_for_production(meta_data, file):                    
    if 'prod_files' not in meta_data.keys():                    
        meta_data['prod_files'] = []                    
    meta_data['prod_files'].append(file)                    
                    
def copy_prod_files(source, target, meta_data):                    
    if 'prod_files' in meta_data.keys():                    
        for file in meta_data['prod_files']:                    
            if not (target/file).exists():                    
                if (source/file).exists():                    
                    shutil.copy(source/file, target/file)
                    
def log_dataframe(df, msg=None):                    
    import io                    
    buffer = io.StringIO()                    
    df.info(buf=buffer)                    
    if msg:                    
        log_text = f'Data frame after {msg}:'                    
    else:                    
        log_text = 'Data frame:'                    
    log_text += '\n\t'+str(df.head(2)).replace('\n','\n\t')                    
    log_text += ('\n\t' + buffer.getvalue().replace('\n','\n\t'))                    
    get_logger().info(log_text)
                    
def scoring_criteria(score_param, problem_type, class_count):                    
    if problem_type == 'classification':                    
        scorer_mapping = {                    
                    'recall':{'binary_class': 'recall', 'multi_class': 'recall_weighted'},                    
                    'precision':{'binary_class': 'precision', 'multi_class': 'precision_weighted'},                    
                    'f1_score':{'binary_class': 'f1', 'multi_class': 'f1_weighted'},                    
                    'roc_auc':{'binary_class': 'roc_auc', 'multi_class': 'roc_auc_ovr_weighted'}                    
                   }                    
        if (score_param.lower() == 'roc_auc') and (class_count > 2):                    
            score_param = make_scorer(roc_auc_score, needs_proba=True,multi_class='ovr',average='weighted')                    
        else:                    
            class_type = 'binary_class' if class_count == 2 else 'multi_class'                    
            if score_param in scorer_mapping.keys():                    
                score_param = scorer_mapping[score_param][class_type]                    
            else:                    
                score_param = 'accuracy'                    
    return score_param

def mlflowSetPath(path, name):                    
    db_name = str(Path(path)/'mlruns')                    
    mlflow.set_tracking_uri('file:///' + db_name)                    
    mlflow.set_experiment(str(Path(path).name))                    


def logMlflow( params, metrices, estimator, algoName=None):                    
    run_id = None                    
    for k,v in params.items():                    
        mlflow.log_param(k, v)                    
    for k,v in metrices.items():                    
        mlflow.log_metric(k, v)                    
    if 'CatBoost' in algoName:                    
        model_info = mlflow.catboost.log_model(estimator, 'model')                    
    else:                    
        model_info = mlflow.sklearn.log_model(sk_model=estimator, artifact_path='model')                    
    mlflow.set_tags({'processed':'no', 'registered':'no'})                    
    if model_info:                    
        run_id = model_info.run_id                    
    return run_id                    

def get_classification_metrices( actual_values, predicted_values):                    
    result = {}                    
    accuracy_score = sklearn.metrics.accuracy_score(actual_values, predicted_values)                    
    avg_precision = sklearn.metrics.precision_score(actual_values, predicted_values,                    
        average='macro')                    
    avg_recall = sklearn.metrics.recall_score(actual_values, predicted_values,                    
        average='macro')                    
    avg_f1 = sklearn.metrics.f1_score(actual_values, predicted_values,                    
        average='macro')                    
                    
    result['accuracy'] = math.floor(accuracy_score*10000)/100                    
    result['precision'] = math.floor(avg_precision*10000)/100                    
    result['recall'] = math.floor(avg_recall*10000)/100                    
    result['f1'] = math.floor(avg_f1*10000)/100                    
    return result                    

        
def validateConfig(base_config):        
    config_file = Path(__file__).parent/'config.json'        
    if not Path(config_file).exists():        
        raise ValueError(f'Config file is missing: {config_file}')        
    config = read_json(config_file)        
    if not base_config['inputPath']:        
        base_config['inputPath'] = config['inputPath']        
        
    if not base_config['outputPath']:        
        base_config['outputPath'] = config['outputPath']        
        
    if not Path(base_config['inputPath']).exists():        
        if not is_file_name_url(base_config['inputPath']):        
            loc = base_config['inputPath']        
            raise ValueError(f'Data location does not exists: {loc}')        
    config = read_json(config_file)        
    return base_config,config
        
def save_model(outputPath, usecase, estimator, features, metrices, params, scoring, meta_data, ml_save=False):        
    if ml_save:        
        # mlflow log model, metrices and parameters        
        mlflowSetPath(str(outputPath.resolve()), usecase)        
        with mlflow.start_run(run_name = model_name):        
            run_id = logMlflow(params, metrices, estimator, model_name.split('_')[0])        
            mlflow.log_text(','.join(features), 'features.txt')        
    else:        
        joblib.dump(estimator, outputPath/output_file['model'])        
        write_json({'scoring_criteria': scoring, 'metrices':metrices, 'param':params}, outputPath/output_file['performance'])        
        add_file_for_production(meta_data, output_file['model'])        
        add_file_for_production(meta_data, output_file['performance'])



def train(base_config):        
    base_config,config = validateConfig(base_config)        
    inputPath = Path(base_config['inputPath'])        
    outputPath = Path(base_config['outputPath'])        
    outputPath.mkdir(parents=True, exist_ok=True)        
    if not outputPath.exists():        
        raise ValueError(f'Target Path not found at {outputPath}')        
    meta_data_file = inputPath/input_file['metaData']        
    if meta_data_file.exists():        
        meta_data = read_json(meta_data_file)        
    else:        
        raise ValueError(f'Configuration file not found: {meta_data_file}')        
    log_file = outputPath/output_file['log']        
    logger = set_logger(log_file)        
    dataLoc = inputPath/input_file['inputData']        
    if not dataLoc.exists():        
        return {'Status':'Failure','Message':'Data location does not exists.'}        
        
    status = dict()        
    usecase = meta_data['usecase']        
    df = pd.read_csv(dataLoc)        
    prev_step_output = meta_data['featureengineering']['Status']

    # split the data for training        
    selected_features = prev_step_output['selected_features']        
    target_feature = config['target_feature']        
    train_features = prev_step_output['total_features'].copy()        
    train_features.remove(target_feature)        
    X_train, X_test, y_train, y_test = train_test_split(df[train_features],df[target_feature],train_size=config['train_ratio'])
    logger.info('Data balancing done')
    
    #select scorer
    scorer = scoring_criteria(config['scoring_criteria'],config['problem_type'], df[target_feature].nunique())
    logger.info('Scoring criterio: accuracy')
    
    #Training model
    logger.info('Training XGBClassifier for modelBased')
    features = selected_features['modelBased']            
    estimator = XGBClassifier()            
    param = config['algorithms']['XGBClassifier']
    grid = RandomizedSearchCV(estimator, param, scoring=scorer, n_iter=config['optimization_param']['iterations'],cv=config['optimization_param']['trainTestCVSplit'])            
    grid.fit(X_train[features], y_train)            
    train_score = grid.best_score_ * 100            
    best_params = grid.best_params_            
    estimator = grid.best_estimator_
    
    #model evaluation
    y_pred = estimator.predict(X_test[features])
    test_score = round(accuracy_score(y_test,y_pred),2) * 100
    logger.info('Confusion Matrix:')
    logger.info('\n' + pd.DataFrame(confusion_matrix(y_test,y_pred)).to_string())
    metrices = get_classification_metrices(y_test,y_pred)
    metrices.update({'train_score': train_score, 'test_score':test_score})
        
    meta_data['training'] = {}        
    meta_data['training']['features'] = features        
    run_id = ''        
    scoring = config['scoring_criteria']        
    save_model(outputPath, usecase, estimator,features, metrices,best_params,scoring,meta_data, base_config['ml'])        
    copy_prod_files(inputPath, outputPath, meta_data)        
    write_json(meta_data,  outputPath/output_file['metaData'])        
        
    # return status        
    status = {'Status':'Success','mlflow_run_id':run_id,'FeaturesUsed':features,'test_score':metrices['test_score'],'train_score':metrices['train_score']}        
    logger.info(f'Test score: {test_score}')        
    logger.info(f'Train score: {train_score}')        
    logger.info(f'MLflow run id: {run_id}')        
    logger.info(f'output: {status}')        
    return json.dumps(status)
        
if __name__ == '__main__':        
    parser = argparse.ArgumentParser()        
    parser.add_argument('-i', '--inputPath', help='path of the input data')        
    parser.add_argument('-u', '--inputUri', help='uri for the input data')        
    parser.add_argument('-o', '--outputPath', help='path for saving the output data')        
    parser.add_argument('-ml', action='store_true', help='to use mlflow or not')        
        
    args = parser.parse_args()        
        
    config = {'inputPath':None, 'outputPath':None}        
        
    if args.inputPath:        
        config['inputPath'] = args.inputPath        
    if args.inputUri: #uri has higher preference than input path        
        config['inputPath'] = args.inputUri        
    if args.outputPath:        
        config['outputPath'] = args.outputPath        
    config['ml'] = args.ml        
        
    try:        
        print(train(config))        
    except Exception as e:        
        status = {'Status':'Failure','Message':str(e)}        
        print(json.dumps(status))        