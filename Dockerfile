FROM python:3.11.7
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install -r /code/requirements.txt
COPY ./loan_norm.csv /code/
COPY ./generate_model.py /code/generate_model.py
COPY ./app /code/app
RUN python generate_model.py
RUN mv xgb_model_v2.pkl app/xgb_model_v2.pkl
RUN rm /code/loan_norm.csv
CMD ["fastapi", "run", "app/main.py", "--port", "80"]