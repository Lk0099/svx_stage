FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Python tools
RUN pip install --upgrade \
    pip \
    setuptools==59.8.0 \
    wheel

# VERY IMPORTANT — install these first
RUN pip install \
    numpy==1.23.5 \
    cython==0.29.36

# Copy requirements
COPY requirements.txt .

# Install remaining packages WITHOUT build isolation
RUN pip install \
    --no-cache-dir \
    --no-build-isolation \
    -r requirements.txt

COPY . .

CMD ["python", "main.py"]
