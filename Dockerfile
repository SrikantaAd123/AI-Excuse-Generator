FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip && pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "intelligent_excuse_generator_app.py", "--server.port=8501", "--server.headless=true"]
