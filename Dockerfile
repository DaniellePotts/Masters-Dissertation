FROM python:3.6

EXPOSE 8080

ADD requirements.txt .
RUN pip install -r requirements.txt

FROM openjdk:7

ADD helloworld.py .
CMD [ "python3", "./helloworld.py" ]
