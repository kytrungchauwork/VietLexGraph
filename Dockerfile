FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements-app.txt ./
RUN pip install --no-cache-dir --upgrade pip==26.2.1 \
    && pip install --no-cache-dir -r requirements-app.txt \
    && useradd --create-home --uid 10001 appuser
COPY src/ ./src/
COPY scripts/prepare_data.py ./scripts/prepare_data.py
COPY app.py config.yaml LICENSE ./
COPY samples/sample_legal_text.txt ./samples/sample_legal_text.txt
RUN mkdir -p models .cache && chown -R appuser:appuser /app
USER appuser
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
