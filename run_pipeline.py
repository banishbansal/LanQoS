
from pathlib import Path
import subprocess
import sys
import json

if len(sys.argv) < 2:
    print('Please provide the data path')
    print('Usage python run_pipeline.py data_path')
    exit()
    
data_path = sys.argv[1]
print(f'Data Location: {data_path}')
cwd = Path(__file__).parent

load_file = str(cwd/'DataIngestion'/'code.py')
transformer_file = str(cwd/'DataTransformation'/'code.py')
selector_file = str(cwd/'FeatureEngineering'/'code.py')
train_folder = cwd
register_file = str(cwd/'ModelRegistry'/'code.py')
deploy_file = str(cwd/'ModelServing'/'code.py')

print('Running dataIngestion')
cmd = ['python', load_file, '-i', data_path]
result = subprocess.check_output(cmd)
result = result.decode('utf-8')
print(result)    
result = json.loads(result[result.find('{"Status":'):])
if result['Status'] == 'Failure':
    exit()

print('Running DataTransformation')
cmd = ['python', transformer_file]
result = subprocess.check_output(cmd)
result = result.decode('utf-8')
print(result)
result = json.loads(result[result.find('{"Status":'):])
if result['Status'] == 'Failure':
    exit()

print('Running FeatureEngineering')
cmd = ['python', selector_file]
result = subprocess.check_output(cmd)
result = result.decode('utf-8')
print(result)
result = json.loads(result[result.find('{"Status":'):])
if result['Status'] == 'Failure':
    exit()

train_models = [f for f in train_folder.iterdir() if 'ModelTraining' in f.name]
for model in train_models:
    print(f'Running {model.name}')
    cmd = ['python', str(model/'code.py')]
    train_result = subprocess.check_output(cmd)
    train_result = train_result.decode('utf-8')
    print(train_result)    

print('Running ModelRegistry')
cmd = ['python', register_file]
result = subprocess.check_output(cmd)
result = result.decode('utf-8')
print(result)
result = json.loads(result[result.find('{"Status":'):])
if result['Status'] == 'Failure':
    exit()

print('Running ModelServing')
cmd = ['python', deploy_file]
result = subprocess.check_output(cmd)
result = result.decode('utf-8')
print(result)
