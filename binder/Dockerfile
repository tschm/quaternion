# Set the base image to jupyter
FROM lobnek/jupyter:v2.5

#FROM jupyter/minimal-notebook:45b8529a6bfc

#RUN conda config --env --add pinned_packages 'openjdk>8.0.121'
USER root
RUN conda install -y -c conda-forge matplotlib && conda clean -y --all
USER $NB_USER

COPY ./work $WORK
