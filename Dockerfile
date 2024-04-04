FROM python:3.11
RUN pip install poetry==1.7.1

COPY ./ /app
WORKDIR /app
RUN poetry check
RUN poetry config virtualenvs.create false
RUN poetry install
EXPOSE 8000
CMD ["poetry", "run","uvicorn", "app:app","--host", "0.0.0.0", "--port", "8000"]
