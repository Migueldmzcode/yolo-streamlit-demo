# Imagen para desplegar la app en AWS (EC2, ECS/Fargate) o en cualquier nube con contenedores
FROM python:3.12-slim

# Librerías del sistema que necesita OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Descarga los pesos en la construcción: el contenedor arranca sin depender de internet
RUN python -c "from ultralytics import YOLO; YOLO('yolo26n.pt'); YOLO('yolo11n.pt')" \
    && mkdir -p models && mv -f yolo26n.pt yolo11n.pt models/ 2>/dev/null || true

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
