FROM python:3.11

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements.txt
RUN python -m pip install --upgrade pip && pip install --resume-retries 5 -r requirements.txt

COPY main.py main.py

COPY models/ models/

CMD ["python", "main.py"]

