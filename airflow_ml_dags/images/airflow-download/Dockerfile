FROM python:3.6-slim-stretch

WORKDIR /app

RUN pip install --no-cache click==7.1.2 pandas scikit-learn==0.24.2

COPY . .

ENTRYPOINT ["python", "download.py"]