# Set the base image to jupyter
FROM lobnek/jupyter:v1.1

#FROM jupyter/minimal-notebook:45b8529a6bfc

#RUN conda config --env --add pinned_packages 'openjdk>8.0.121'
RUN conda install -y -c conda-forge matplotlib && conda clean -y --all

COPY ./work /home/${NB_USER}/work
