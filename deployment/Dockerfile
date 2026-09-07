FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt pyproject.toml README.md ./
COPY minillm ./minillm
COPY deployment ./deployment
RUN pip install --no-cache-dir .[api]
ENV MINILLM_CONFIG=/model/config.json \
    MINILLM_CHECKPOINT=/model/model.pt \
    MINILLM_TOKENIZER=/model/tokenizer.json
EXPOSE 8000
CMD ["uvicorn", "deployment.api:app", "--host", "0.0.0.0", "--port", "8000"]
