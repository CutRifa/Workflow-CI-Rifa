# Gunakan Python versi kecil
FROM python:3.9-slim

# Set folder kerja di dalam container
WORKDIR /app

# Salin semua file proyek ke dalam container
COPY . /app

# Install library yang dibutuhkan (sesuaikan dengan proyek Anda)
RUN pip install --no-cache-dir mlflow pandas scikit-learn requests

# Perintah yang jalan saat container hidup
# (Arahkan ke script python utama Anda)
CMD ["python", "MLProject/modelling.py"]