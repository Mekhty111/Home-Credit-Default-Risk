# Image for the scoring web app. Works on Heroku (heroku.yml), Azure, or any Docker host.
# Listens on $PORT (default 8000). UI at /ui, API docs at /docs.
FROM python:3.11-slim

# LightGBM needs the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
ENV PYTHONUNBUFFERED=1

WORKDIR /home/user/credit

COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY --chown=user app/ app/
COPY --chown=user models/*.pkl models/*.json models/scorecard_table.csv models/

USER user
EXPOSE 8000
CMD uvicorn main:app --app-dir app --host 0.0.0.0 --port ${PORT:-8000}
