FROM python:3.11
RUN pip install "dvc[gdrive]"
RUN pip install poetry==1.7.1

COPY ./ /app
WORKDIR /app

# initialise dvc
RUN dvc init --no-scm -f
# configuring remote server in dvc
RUN dvc remote add -d storage gdrive://1LWnLHric6xXZk6g51eppbGWFZYG_5yF4
RUN dvc remote modify storage gdrive_use_service_account true
RUN dvc remote modify storage gdrive_service_account_json_file_path creds.json

#RUN dvc add models/epoch=2-step=201.ckpt

# pulling the trained model
RUN dvc pull models/epoch=2-step=201.ckpt.dvc

RUN poetry check
RUN poetry config virtualenvs.create false
RUN poetry install
EXPOSE 8000
CMD ["poetry", "run","uvicorn", "app:app","--host", "0.0.0.0", "--port", "8000"]
