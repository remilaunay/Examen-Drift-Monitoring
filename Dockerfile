FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN groupadd --gid 1000 app && useradd --uid 1000 --gid app --create-home app
COPY --chown=1000:1000 src ./src

USER 1000:1000

CMD ["python", "src/main.py"]
