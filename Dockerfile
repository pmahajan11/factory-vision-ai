FROM python:3.9-slim

RUN apt-get update

WORKDIR /usr/src/FactoryVisionAI

COPY requirements.txt ./

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 80

CMD ["python3", "app/app.py"]