FROM python:3.8
COPY . /usr/src/appbdconexion
WORKDIR /usr/src/appbdconexion

RUN pip install -r requirements.txt
RUN pip install psycopg2
RUN pip install requests
RUN pip install holidays
RUN pip install numpy

ENTRYPOINT uvicorn --host 0.0.0.0 main:app --reload --port 8001
