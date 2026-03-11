FROM python:3.11-slim

WORKDIR /app

# Copy project
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 8000

# Run the application
CMD ["uvicorn", "backend.main_api:app", "--host", "0.0.0.0", "--port", "8000"]
