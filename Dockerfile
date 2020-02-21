# pull base image
FROM python:3.7-slim
#add user
RUN adduser  textclassifier

# [Start app environment]
WORKDIR /home/textclassifier
COPY requirements.txt requirements.txt
COPY app app
COPY model model
COPY textclassifier.py config.py  ./
ENV FLASK_APP textclassifier.py
RUN python -m venv venv  
RUN venv/bin/pip install --upgrade pip  
#[end app environement]

#[start install]
RUN venv/bin/pip install  -r requirements.txt
#[end install]

RUN chown -R textclassifier:textclassifier ./
USER textclassifier

EXPOSE 5000
ENTRYPOINT ["venv/bin/python", "textclassifier.py"]
