FROM python:3.7

EXPOSE 8080

ADD requirements.txt .
RUN pip3 install --upgrade minerl
RUN pip3 install -r requirements.txt

# FROM openjdk:7

ADD interact.py .
ADD functions/* .
CMD [ "python", "interact.py" ]
