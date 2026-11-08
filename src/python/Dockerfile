FROM python:3.12-slim

WORKDIR /app

COPY src/python/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src/python/main.py ./main.py
ENV TITANIC_DATA_DIR=/app/data

CMD ["python", "main.py"]