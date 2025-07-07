FROM python:3.11-slim

WORKDIR /app

COPY apps/ .

COPY models/vectorizer.pkl /app/models/vectorizer.pkl

RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader stopwords wordnet

EXPOSE 8501

CMD ["gunicorn", "--bind", "0.0.0.0:8501", "--timeout", "120", "app:app"]