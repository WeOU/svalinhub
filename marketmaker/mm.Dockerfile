FROM python:3.6.3-stretch

RUN mkdir /var/lib/app
WORKDIR /var/lib/app
COPY requirements_mm.txt .
RUN pip install -r requirements_mm.txt
COPY app/marketmaker.py ./
CMD ["/bin/bash", "-c", "python marketmaker.py", ">> /logs/mm.log 2>&1"]
