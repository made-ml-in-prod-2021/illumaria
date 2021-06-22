FROM python:3.6-slim-stretch

COPY train.py /train.py
COPY requirements.txt /requirements.txt
RUN pip3 install --no-cache -r requirements.txt

ENTRYPOINT ["python", "train.py"]