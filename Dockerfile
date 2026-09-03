FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Model slices are generated at first startup (server auto-runs splitter if missing)
CMD ["python", "-m", "server.main"]
