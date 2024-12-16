# SKILLMATCH.AI: NOVA SBE workproject
This repo cotains the source code for **Skillmatch.ai**, a web application that operationalizes the skills-first approach in the labour market. It integrates functionalities such as resume screening, tailored interview questions, bespoke apology letters, and role similarity analysis, offering a comprehensive solution for skills-based hiring and career development.  Built with a scalable architecture, the application addresses recruitment inefficiencies and skill gaps. Although limitations such as scalability constraints, reliance on external APIs, and the need for real-world data remain, future iterations aim to refine infrastructure, enhance usability, and expand applicability, advancing the adoption of skills-first practices across industries.

## Instructions:

### Setting up the conda venv

Create the environment with the config-file
```bash
conda env create -f environment.yml
```
Conda should now have set you up with all the necessary dependencies to run the project. Please activate the environment:
```bash
conda activate workproject
```
You could also enable a jupyter-kernel if you want to use this environment in jupyter notebooks:
```bash
conda python -m ipykernel install --user --name workproject --display-name "workproject"
```

### Run the we application locally
#### To run the flask application locally on your machine (first navigate to the user_interface directory)
```bash
flask run
```
### Download and run the docker image

1. If not already installed, install *Docker* for your OS
2. Log in to docker in terminal
```bash
docker login
```
1. Pull this docker image (might need sudo first)
```bash
docker pull victorbjorsvik/skillmatch_ai:latest
```
1. Run the container (might need sudo first)
```bash
docker run - 5000:5000 victorbjorsvik/skillmatch_ai:latest
```
or if you want to add the environment variable for using the admin user:
```bash
docker run -5000:5000 -e OPENAI_API_KEY="<your openai api key>" victorbjorsvik/skillmatch_ai:latest
```

These steps should sucsessfully host an instance of the webapp which you can interact with through your browser.

## Overview of the application
### File tree: 
![img](pictures\file_tree.png)
### Home page:![img](pictures\index.jpg)
### External Recruiting (before analysis):
 ![img](pictures\ext_before.jpg)
### External Recruiting (after analysis): 
![img](pictures\ext_after.jpg)
### Tailored Questions (before analysis): 
![img](pictures\quest_before.jpg)
### Tailored Questions (after analysis):
 ![img](pictures\quest_after.jpg)
### Bespoke Apologies (before analysis):
 ![img](pictures\bespoke_before.jpg)
### Bespoke Apologies (after analysis): 
![img](pictures\quest_after.png)
### Similar Roles (before analysis):
![img](pictures\roles_before.jpg)
### Similar Roles (after analyis):
![img](pictures\roles_after.jpg)



## Collaborators
* Hanna Borchgrevink Pedersen
* Tim Gunkel
* Irene Abbatelli
* Luca Oeztekin

### Acknowledgements
Big thanks to Amira for letting us use [her code](https://github.com/amiradridi/Job-Resume-Matching) as a building block for our project:
