# NOVA SBE workproject matching_algo
> This Repo contains the matching algortithm between candidate and job description for our workproject at NOVA SBE

### Broad overview of the workflow and pipeline:
* Datasets: {job descriptions, CVs}
* Tokenization for both
* Extract relevant skills using Spacy PhraseMatcher etc.
* Matching rules (big job – dependent on which datasets we are using)
* Embeddings (skill2Vec, BERT, etc.)
* Calculate similarity scores and rankings


### Next steps:
1. Job description (Curation, Cleaning, Tokenization)
2. CVs (Curation, Cleaning, Tokenization)
3. Extract relevant skills using Spacy PhraseMatcher etc.
4. Copy / refine matching rules from previous work
5. Embeddings (skill2Vec, BERT, etc.)


### CAVEATS:
 * Bias in models
 * Tailor interview questions to the pool of applicants
 * Measure of performance (e.g. labelled datasets)
 * Productivity data for employees



### Setting up the conda venv
First create a new conda environment for the project (e.g.)
```bash
conda create -n workproject
```
Then activate the environment
```bash
conda activate workproject
```
Lastly config the environment with the config-file
```bash
conda env create -f environment.yml
```
Conda should now have set you up with all the necessary dependencies to run the project.

### Usage
#### To run the flask application locally on your machine (first navigate to the user_interface directory)
```bash
flask run
```
(or for debug mode)
```bash
python app.py
```

### Acknowledgements
Big thanks to Amira for letting us use [her code](https://github.com/amiradridi/Job-Resume-Matching) as a building block for our project:
