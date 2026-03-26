FROM python:3.13.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app/ ./app/
COPY model/ ./model/

# Expose ports: 8000 = FastAPI, 8501 = Streamlit
EXPOSE 8000 8501

# Start both services using a shell script
COPY start.sh .
RUN chmod +x start.sh

CMD ["./start.sh"]