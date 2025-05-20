# use offical python image
FROM python:3.11-slim

# set the working directory
WORKDIR /app
# copy the current directory contents into the container at /app
COPY . /app 

# install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# make port 500 available to the world outside this container
EXPOSE 5000
# Run app.py when the container launches 
CMD gunicorn --workers=4 --bind 0.0.0.0:5000 app:app
