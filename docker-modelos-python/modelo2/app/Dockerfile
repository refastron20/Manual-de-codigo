FROM python:3.8
COPY . usr/src/app
WORKDIR /usr/src/app

RUN pip install -r requirements.txt
RUN pip install numpy
RUN pip install pandas
RUN pip install sklearn
RUN pip install scipy
RUN pip install requests
RUN pip install psycopg2


ENTRYPOINT uvicorn --host 0.0.0.0 main:app --reload --port 8007
