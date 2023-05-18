FROM python:3.8-slim
WORKDIR /webapp
COPY . .
RUN rm -r ./docker
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5002
CMD [ "gunicorn", "app:app",  "--timeout", "120", "--workers=3", "--threads=3", "--worker-connections=1000"]