FROM python:3.6-slim-stretch

COPY download.py /download.py
COPY requirements.txt /requirements.txt
RUN pip3 install --no-cache -r requirements.txt

ENTRYPOINT ["python", "download.py"]