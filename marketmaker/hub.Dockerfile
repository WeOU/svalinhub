FROM python:3.6.3-stretch

EXPOSE 8000
RUN mkdir /var/lib/app
WORKDIR /var/lib/app
COPY requirements_hub.txt .
RUN pip install -r requirements_hub.txt
COPY app/hub.py app/
COPY app/marketmaker.py app/
CMD ["/bin/bash", "-c", "gunicorn --bind 0.0.0.0:8000 --workers 1 app.hub:__hug_wsgi__", ">> /logs/hub.log 2>&1"]
