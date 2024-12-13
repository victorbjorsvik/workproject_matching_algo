from spacy.lang.en import English


class JobInfoExtraction:
    """
    A class to extract job-related skills information using spaCy's EntityRuler.

    Attributes:
        skills_patterns_path (str): Path to the JSON file containing skill patterns.
        nlp (Language): spaCy language model with the EntityRuler pipeline.
    """

    def __init__(self, skills_patterns_path):
        """
        Initializes the JobInfoExtraction class with the specified skills patterns.

        Args:
            skills_patterns_path (str): Path to the JSON file with skill patterns.
        """
        self.skills_patterns_path = skills_patterns_path
        self.nlp = English()
        # Add EntityRuler to the spaCy pipeline and load patterns from the specified path
        ruler = self.nlp.add_pipe("entity_ruler")
        ruler.from_disk(self.skills_patterns_path)

    def match_skills_by_spacy(self, job, display=False):
        """
        Extracts skills mentioned in a job description using spaCy.

        Args:
            job (str): Job description text.
            display (bool): Whether to print the extracted skills (default: False).

        Returns:
            list: A list of unique skill labels extracted from the job description.
        """
        doc = self.nlp(job)  # Process the job description
        job_skills = []

        for ent in doc.ents:
            labels_parts = ent.label_.split('|')
            # Check if the label indicates a skill
            if labels_parts[0] == 'SKILL':
                skill_name = labels_parts[1].replace('-', ' ')
                if display:
                    print((ent.text, ent.label_))
                # Add the skill to the list if not already present
                if skill_name not in job_skills:
                    job_skills.append(skill_name)

        return job_skills

    def extract_entities(self, jobs):
        """
        Processes a DataFrame of job descriptions and extracts skills.

        Args:
            jobs (DataFrame): A pandas DataFrame with a column named 'raw' containing job descriptions.

        Returns:
            DataFrame: The input DataFrame with an additional 'Skills' column listing extracted skills.
        """
        # Clean the job descriptions and extract skills
        jobs['Skills'] = (
            jobs['raw']
            .str.replace('.', ' ')
            .str.replace('\n', ' ')
            .apply(self.match_skills_by_spacy)
        )
        return jobs
