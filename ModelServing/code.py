
from http.server import BaseHTTPRequestHandler,HTTPServer        
from socketserver import ThreadingMixIn        
import os        
from os.path import expanduser        
import platform        
import threading        
import subprocess        
import argparse        
import re        
import cgi        
import json        
import sys        
from pathlib import Path        
from predict import deploy
from groundtruth import groundtruth       
config_input = None        
input_file = {
    "inputData": "rawData.dat",
    "metaData": "modelMetaData.json"
}

def read_json(file_path):        
    data = None        
    with open(file_path,'r') as f:        
        data = json.load(f)        
    return data        
        
class HTTPRequestHandler(BaseHTTPRequestHandler):        
        
	def do_POST(self):        
		print('PYTHON ######## REQUEST ####### STARTED')        
		if None != re.search('/AION/', self.path):        
			ctype, pdict = cgi.parse_header(self.headers.get('content-type'))        
			if ctype == 'application/json':        
				length = int(self.headers.get('content-length'))        
				data = self.rfile.read(length)        
				operation = self.path.split('/')[-1]        
				data = json.loads(data)        
				dataStr = json.dumps(data)        
				if operation.lower() == 'predict':        
					deployobj = deploy(config_input)        
					output=deployobj.predict(dataStr)        
					resp = output        
				elif operation.lower() == 'groundtruth':
					gtObj = groundtruth(config_input)				
					output = gtObj.actual(dataStr)
					resp = output        
				else:        
					outputStr = json.dumps({'Status':'Error','Msg':'Operation not supported'})        
					resp = outputStr        
        
			else:        
				outputStr = json.dumps({'Status':'ERROR','Msg':'Content-Type Not Present'})        
				resp = outputStr        
			resp=resp+'\n'        
			resp=resp.encode()        
			self.send_response(200)        
			self.send_header('Content-Type', 'application/json')        
			self.end_headers()        
			self.wfile.write(resp)        
		else:        
			print('python ==> else1')        
			self.send_response(403)        
			self.send_header('Content-Type', 'application/json')        
			self.end_headers()        
			print('PYTHON ######## REQUEST ####### ENDED')        
		return        
        
	def do_GET(self):        
		print('PYTHON ######## REQUEST ####### STARTED')        
		if None != re.search('/AION/', self.path):        
			self.send_response(200)        
			self.send_header('Content-Type', 'application/json')        
			self.end_headers()        
			meta_data_file = config_input['inputPath']/input_file['metaData']        
			if meta_data_file.exists():        
				meta_data = read_json(meta_data_file)        
			else:        
				raise ValueError(f'Configuration file not found: {meta_data_file}')
			features = meta_data['load_data']['selected_features']
			bodydes='['
			for x in features:
				if bodydes != '[':
					bodydes = bodydes+','
				bodydes = bodydes+'{"'+x+'":"value"}'	
			bodydes+=']'
			urltext = 'http://'+config_input['ipAddress']+':'+str(config_input['portNo'])+'/AION/predict'
			msg="""
URL:{url}
RequestType: POST
Content-Type=application/json
Body: {displaymsg}
Output: prediction,probability(if Applicable),remarks corresponding to each row. 
			""".format(url=urltext,displaymsg=bodydes)        
			self.wfile.write(msg.encode())        
		else:        
			self.send_response(403)        
			self.send_header('Content-Type', 'application/json')        
			self.end_headers()        
		return        
        
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):        
	allow_reuse_address = True        
        
	def shutdown(self):        
		self.socket.close()        
		HTTPServer.shutdown(self)        
        
class SimpleHttpServer():        
	def __init__(self, ip, port):        
		self.server = ThreadedHTTPServer((ip,port), HTTPRequestHandler)        
        
	def start(self):        
		self.server_thread = threading.Thread(target=self.server.serve_forever)        
		self.server_thread.daemon = True        
		self.server_thread.start()        
        
	def waitForThread(self):        
		self.server_thread.join()        
        
	def stop(self):        
		self.server.shutdown()        
		self.waitForThread()        
        
if __name__=='__main__':        
	parser = argparse.ArgumentParser(description='HTTP Server')        
	parser.add_argument('-i', '--inputPath', help='path of the input data')        
	parser.add_argument('-o', '--outputPath', help='path of the input data')	
	parser.add_argument('-ip','--ipAddress', help='HTTP Server IP')        
	parser.add_argument('-p','--portNo', type=int, help='Listening port for HTTP Server')        
	args = parser.parse_args()        
	config_file = Path(__file__).parent/'config.json'        
	if not Path(config_file).exists():        
		raise ValueError(f'Config file is missing: {config_file}')        
	config = read_json(config_file)        
	if args.inputPath:        
		config['inputPath'] = args.inputPath
	if args.outputPath:        
		config['outputPath'] = args.outputPath        
	if args.ipAddress:        
		config['ipAddress'] = args.ipAddress        
	if args.portNo:        
		config['portNo'] = args.portNo        
	config['inputPath'] = Path(config['inputPath'])/'production'        
	server = SimpleHttpServer(config['ipAddress'],int(config['portNo']))        
	config_input = config        
	print('HTTP Server Running...........')  
	print('For Prediction')
	print('================')
	print('Request Type: Post')
	print('Content-Type: application/json')	
	print('URL:http://'+config['ipAddress']+':'+str(config['portNo'])+'/AION/predict')	
	print('\nFor GroundTruth')
	print('================')
	print('Request Type: Post')
	print('Content-Type: application/json')	
	print('URL:http://'+config['ipAddress']+':'+str(config['portNo'])+'/AION/groundtruth')	
	print('\nFor Help')
	print('================')
	print('Request Type: Get')
	print('Content-Type: application/json')	
	print('URL:http://'+config['ipAddress']+':'+str(config['portNo'])+'/AION/Help')	
	server.start()        
	server.waitForThread()
