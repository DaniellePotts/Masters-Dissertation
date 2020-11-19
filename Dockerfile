FROM python:3.7

EXPOSE 8080

ADD requirements.txt .
RUN pip3 install --upgrade minerl
RUN pip3 install -r requirements.txt
RUN apt-get update ##[edited]
RUN apt-get install 'ffmpeg'\
    'libsm6'\ 
    'libxext6'  -y
RUN mkdir 'functions'
RUN mkdir 'resources-actions'
RUN mkdir 'models'
# FROM openjdk:7

ADD interact.py .
ADD functions/ActionCombos.py /functions
ADD functions/interact_with_env.py /functions
ADD functions/Utils.py /functions
ADD functions/DQfDModel.py /functions
ADD models/*.h5 /models
ADD resources-actions/*.sav /resources-actions

CMD [ "python", "interact.py" ]
