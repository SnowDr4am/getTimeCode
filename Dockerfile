FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models

RUN apt-get -o Acquire::Retries=10 update \
    && apt-get -o Acquire::Retries=10 install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY *.py pytest.ini ./
COPY tests ./tests

CMD ["python", "bot.py"]
