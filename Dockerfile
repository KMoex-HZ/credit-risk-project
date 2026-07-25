FROM python:3.11-slim

WORKDIR /app

# Copy requirements dulu (biar Docker cache layer ini, install ga perlu ulang tiap kali kode berubah)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua source code yang dibutuhin API
COPY api/ ./api/
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]