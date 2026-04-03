FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY uv.lock pyproject.toml ./
RUN pip install uv && uv sync --frozen --no-install-project

COPY . .

RUN uv pip install gitpython

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 4000

CMD ["python", "-m", "uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "4000"]
