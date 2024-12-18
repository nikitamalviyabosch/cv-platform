FROM python:3.10

RUN apt-get update && apt-get install -y libgl1-mesa-glx

WORKDIR /app
COPY resources/ resources/
COPY app.py .
COPY config.cfg .
COPY dataset.py .
COPY Dockerfile .
COPY requirements.txt .
COPY train.py .
COPY utility.py .

# COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
RUN anomalib install -v

EXPOSE 3002

CMD python ./app.py


