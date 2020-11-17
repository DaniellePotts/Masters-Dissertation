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
RUN mkdir 'resources'
RUN mkdir 'models'
# FROM openjdk:7

ADD interact.py .
ADD functions/ActionCombos.py /functions
ADD functions/interact_with_env.py /functions
ADD functions/Utils.py /functions
ADD functions/DQfDModel.py /functions
ADD training/expert_model_1603572051_749999.h5 /models
ADD resources/action_combos_treechop.sav /resources
ADD resources/unique_angles_treechop.sav /resources

CMD [ "python", "interact.py" ]
