# Public demo: recorded-run viewer + rule-check playground. Standard library only.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY agent/ agent/
COPY web/ web/
COPY data/recorded_run.jsonl data/recorded_run.jsonl
RUN useradd --create-home --uid 10001 app
USER app
CMD ["python", "-m", "web.app"]
