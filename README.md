# Detector de objetos con YOLO + Streamlit

**Master Class CESUMA · Tema 10: Ecosistemas en la nube y puesta en producción de sistemas de IA**
Dr. Miguel Angel Domínguez González

Aplicación web en la que el usuario sube una imagen y un modelo YOLO preentrenado (COCO, 80 clases) detecta los objetos presentes, los dibuja sobre la imagen y muestra una tabla con la clase, la confianza y la caja de cada detección.

---

## 1. Estructura del repositorio

```
yolo-streamlit-demo/
├── app.py                 # Interfaz + inferencia (Streamlit + Ultralytics)
├── requirements.txt       # Dependencias de Python (PyTorch solo-CPU)
├── packages.txt           # Dependencias del sistema (apt) para OpenCV
├── .streamlit/config.toml # Límite de subida (10 MB) y tema visual
├── models/                # Pesos versionados: yolo26n.pt, yolo11n.pt (~5.5 MB c/u)
├── samples/               # Imágenes de prueba
├── Dockerfile             # Contenedor para AWS u otra nube
└── README.md
```

> **Buena práctica de producción:** los pesos del modelo se versionan junto con el código (carpeta `models/`). Así cada versión de la app queda ligada a una versión exacta del modelo y se puede hacer *rollback* con Git.

---

## 2. Probar en local (siempre primero)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py               # abre http://localhost:8501
```

Verifica: sube `samples/bus.jpg` → deben aparecer 1 bus y 4 personas.

---

## 3. Ruta A — Streamlit Community Cloud (gratis, ideal para clase y prototipos)

1. **Crear el repositorio en GitHub** (público o privado) y subir todos los archivos:
   ```bash
   git init
   git add .
   git commit -m "Detector YOLO v1.0.0"
   git branch -M main
   git remote add origin https://github.com/<usuario>/yolo-streamlit-demo.git
   git push -u origin main
   ```
2. Entrar a **https://share.streamlit.io** e iniciar sesión con GitHub.
3. **Create app → Deploy a public app from GitHub**: repositorio, rama `main`, archivo `app.py`.
4. En **Advanced settings** elegir **Python 3.12** (y, si hubiera, capturar *secrets*; esta app no los necesita).
5. **Deploy.** La plataforma instala `packages.txt` (apt) y después `requirements.txt` (pip). La primera construcción tarda unos minutos por PyTorch.
6. Revisar **Manage app → Logs**: deben aparecer las líneas `Cargando modelo…` e `inferencia modelo=… latencia_ms=…`.
7. Compartir la URL `https://<nombre>.streamlit.app`.
8. **Actualizar**: cada `git push` a `main` redespliega automáticamente (CI/CD básico).

**Límites a considerar** (documentación de Streamlit, aprox.): hasta ~2.7 GB de RAM y 2 núcleos compartidos; la app se duerme tras 12 h sin tráfico. Por eso se usa PyTorch solo-CPU, un modelo *nano* y `@st.cache_resource`.

---

## 4. Ruta B — AWS EC2 + Docker (control total, nivel intermedio)

```bash
# En tu equipo (o en la propia instancia)
docker build -t yolo-app:1.0.0 .
docker run -p 8501:8501 yolo-app:1.0.0       # prueba local

# (Opcional) publicar la imagen en Amazon ECR
aws ecr create-repository --repository-name yolo-app
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <cuenta>.dkr.ecr.us-east-1.amazonaws.com
docker tag yolo-app:1.0.0 <cuenta>.dkr.ecr.us-east-1.amazonaws.com/yolo-app:1.0.0
docker push <cuenta>.dkr.ecr.us-east-1.amazonaws.com/yolo-app:1.0.0
```

1. Lanzar una instancia **EC2** (Ubuntu 24.04, `t3.medium` o superior: 2 vCPU / 4 GB).
2. **Security Group**: abrir el puerto 22 (solo tu IP) y 80/443 (público).
3. Instalar Docker en la instancia y ejecutar:
   ```bash
   docker run -d --restart always -p 80:8501 --name yolo <imagen>:1.0.0
   ```
4. Abrir `http://<IP-pública>`. Para producción real: dominio + HTTPS (Nginx/Caddy o un Application Load Balancer con certificado de ACM).
5. **Monitorizar**: `docker logs -f yolo` o enviar logs a **Amazon CloudWatch**; crear alarmas de CPU y de estado.
6. **Costos**: apagar la instancia cuando no se use y configurar un *AWS Budget* con alerta.

---

## 5. Ruta C — AWS gestionado y escalable (nivel avanzado)

- **Amazon ECS (Fargate / Express Mode)**: misma imagen Docker desde ECR; AWS administra servidores, balanceador de carga y autoescalado por número de peticiones o CPU.
  *Nota:* AWS App Runner dejó de aceptar clientes nuevos a partir del 30 de abril de 2026; AWS recomienda ECS Express Mode como alternativa.
- **Amazon SageMaker AI endpoint**: separar el modelo de la interfaz. Streamlit (en ECS o EC2) envía la imagen al endpoint con `boto3` (`invoke_endpoint`); SageMaker gestiona versiones del modelo, autoescalado, *Model Monitor* y despliegues A/B o *canary*.

---

## 6. Checklist de puesta en producción

| Concepto del tema | Cómo se aplica en esta app |
|---|---|
| Servidor de modelo | `@st.cache_resource` carga el modelo una vez por proceso |
| Concurrencia | Streamlit atiende varias sesiones; para más carga: réplicas detrás de un balanceador (ECS) |
| Batching | No aplica a una imagen por petición; con alto volumen → Triton / SageMaker |
| Versionado | `APP_VERSION`, pesos en `models/`, *tags* de Git e imágenes Docker `:1.0.0` |
| Training-serving skew | Mismo preprocesamiento siempre (`preparar_imagen`: EXIF, RGB, tamaño máximo) |
| Subpoblaciones | Evaluar con fotos nocturnas, de celular, de distintos contextos antes de liberar |
| Monitorización | Logs de latencia y número de objetos por petición |
| Seguridad | Límite de 10 MB, validación del archivo, *secrets* fuera del código |
| Licencia | Ultralytics YOLO es **AGPL-3.0**: uso académico libre; uso comercial cerrado requiere licencia Enterprise |

---

## 7. Errores frecuentes

| Síntoma | Solución |
|---|---|
| `ImportError: libGL.so.1` | `packages.txt` con `libgl1` y `libglib2.0-0` |
| "This app has gone over its resource limits" | PyTorch CPU, modelo *nano*, `st.cache_resource`, reducir tamaño de imagen |
| La construcción tarda o falla por PyTorch | Mantener `--extra-index-url .../whl/cpu`; elegir Python 3.12 |
| Primera inferencia lenta | Es el "calentamiento" del modelo; las siguientes son más rápidas |
| La app "duerme" | Normal tras 12 h sin uso en Community Cloud; se reactiva con un clic |
