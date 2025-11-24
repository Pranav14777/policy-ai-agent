# ---- Base Image ----
FROM python:3.10-slim

# ---- Set Workdir ----
WORKDIR /app

# ---- Copy project files ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ---- Expose FastAPI port ----
EXPOSE 8000

# ---- Run the FastAPI server ----
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
