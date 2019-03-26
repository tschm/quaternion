# Set the base image to jupyter
FROM lobnek/jupyter:v2.6

# I am reluctant to use conda install here, as conda updates tornado which will break beakerx
RUN pip install --no-cache-dir matplotlib==3.0.3

COPY ./work $WORK
