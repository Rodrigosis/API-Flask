FROM python:3.9 AS base

WORKDIR /usr/src/app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

ENV FLASK_APP wsgi.py
ENV FLASK_RUN_HOST 0.0.0.0
ENV FLASK_RUN_PORT=80
ENV GUNICORN_WORKERS=1
ENV GUNICORN_THREADS=5
ENV GUNICORN_TIMEOUT=10

EXPOSE ${FLASK_RUN_PORT}

CMD gunicorn \
--bind :${FLASK_RUN_PORT} \
--workers ${GUNICORN_WORKERS} \
--threads ${GUNICORN_THREADS} \
--timeout ${GUNICORN_TIMEOUT} \
--access-logfile - \
"wine:create_app()"
