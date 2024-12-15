FROM continuumio/miniconda3:latest

# Copy environment.yml into the container
COPY environment_linux.yml /tmp/environment_linux.yml

# Create the conda environment
RUN conda env create -f /tmp/environment_linux.yml && conda clean -afy

# Make sure conda is initialized for the shell
# If using bash:
SHELL ["/bin/bash", "-c"]

# Activate the environment and ensure that subsequent commands run with it
# By default, RUN commands start a new shell, so we must combine them.
RUN echo "conda activate workproject" >> ~/.bashrc
ENV PATH=/opt/conda/envs/workproject/bin:$PATH

# Copy your app code into the container
WORKDIR /app
COPY . /app

# Set the working directory to the location of your Flask app
WORKDIR /app/user_interface

# Expose the Flask port
EXPOSE 5000

# Command to run your Flask app
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]

