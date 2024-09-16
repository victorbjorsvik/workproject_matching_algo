# NOVA SBE workproject matching_algo
> This Repo contains the matching algortithm between candidate and job description for our workproject at NOVA SBE

### Broad overview of the workflow and pipeline:
* Datasets: {job descriptions, CVs}
* Tokenization for both
* Extract relevant skills using Spacy PhraseMatcher etc.
* Matching rules (big job – dependent on which datasets we are using)
* Embeddings (skill2Vec, BERT, etc.)
* Calculate similarity scores and rankings