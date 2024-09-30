# Usa una imagen base de Python
FROM python:3.12.5

# Establece el directorio de trabajo
WORKDIR /app

# Copia el archivo de requerimientos y el código de la aplicación
COPY requirements.txt .

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto de tu código
COPY . .

# Expone el puerto en el que la aplicación va a correr
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]