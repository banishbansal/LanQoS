FROM python:3.8-slim-buster
WORKDIR /aion
COPY  requirements.txt requirements.txt
COPY  config.json config.json
COPY  code.py code.py
COPY  predict.py predict.py
COPY  groundtruth.py groundtruth.py
RUN pip install --no-cache-dir -r requirements.txt
ENTRYPOINT ["python", "/aion/code.py","-ip","0.0.0.0","-p","8094"]
