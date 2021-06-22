FROM python:3.6-slim-stretch

COPY split.py /split.py
COPY requirements.txt /requirements.txt
RUN pip3 install --no-cache -r requirements.txt

ENTRYPOINT ["python", "split.py"]