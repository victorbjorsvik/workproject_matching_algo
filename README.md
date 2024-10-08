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



### Installation
```bash
pip install -r requirements.txt
```

### Usage
```bash
python main.py
```

### Acknowledgements
Big thanks to Amira for letting us use [her code](https://github.com/amiradridi/Job-Resume-Matching) as a building block for our project:

