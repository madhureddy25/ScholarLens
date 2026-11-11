# API image. Detection (Lens) needs the ML extras: build with --build-arg EXTRAS=api,anthropic,openai,ml
FROM python:3.12-slim
ARG EXTRAS=api,anthropic,openai
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[${EXTRAS}]"
RUN useradd --create-home scholarlens
USER scholarlens
ENV SCHOLARLENS_SIMILARITY=""
EXPOSE 8000
CMD ["uvicorn", "scholarlens.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
