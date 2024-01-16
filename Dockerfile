# syntax = docker/dockerfile:1
FROM python:3.8-slim-bookworm

LABEL maintainer="Lally Elias <lallyelias87@gmail.com>" \
  vendor="lykmapipo"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=42 \
    RANDOM_SEED=42

WORKDIR /python-joblib-cookbook

COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

VOLUME [ \
    "/python-joblib-cookbook/data", \
    "/python-joblib-cookbook/scripts", \
    "/python-joblib-cookbook/tmp" \
]

CMD ["python"]
