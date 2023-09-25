FROM python:3.10-slim

WORKDIR /backend

COPY requirements.txt /backend/requirements.txt

RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 8000

COPY . ./backend

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]


