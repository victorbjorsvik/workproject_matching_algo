from spacy.lang.en import English


class ResumeInfoExtraction:
    """
    A class to extract skills information from resumes using spaCy's EntityRuler.

    Attributes:
        skills_patterns_path (str): Path to the JSON file containing skill patterns.
        name (str): Name or identifier for this instance of extraction.
        nlp (Language): spaCy language model with the EntityRuler pipeline.
    """

    def __init__(self, skills_patterns_path, name):
        """
        Initializes the ResumeInfoExtraction class with the specified skills patterns and name.

        Args:
            skills_patterns_path (str): Path to the JSON file with skill patterns.
            name (str): Name or identifier for this extraction process.
        """
        self.name = name
        self.skills_patterns_path = skills_patterns_path
        self.nlp = English()
        # Add EntityRuler to the spaCy pipeline and load patterns from the specified path
        ruler = self.nlp.add_pipe("entity_ruler")
        ruler.from_disk(self.skills_patterns_path)

    def match_skills_by_spacy(self, resume, display=False):
        """
        Extracts skills mentioned in a resume using spaCy.

        Args:
            resume (str): Resume text.
            display (bool): Whether to print the extracted skills (default: False).

        Returns:
            list: A list of unique skill labels extracted from the resume.
        """
        doc = self.nlp(resume)  # Process the resume text
        resume_skills = []

        for ent in doc.ents:
            labels_parts = ent.label_.split('|')
            # Check if the label indicates a skill
            if labels_parts[0] == 'SKILL':
                skill_name = labels_parts[1].replace('-', ' ')
                if display:
                    print((ent.text, ent.label_))
                # Add the skill to the list if not already present
                if skill_name not in resume_skills:
                    resume_skills.append(skill_name)

        return resume_skills

    def extract_entities(self, resumes):
        """
        Processes a DataFrame of resumes and extracts skills.

        Args:
            resumes (DataFrame): A pandas DataFrame with a column named 'raw' containing resume texts.

        Returns:
            DataFrame: The input DataFrame with an additional 'Skills' column listing extracted skills.
        """
        # Clean the resume texts and extract skills
        resumes['Skills'] = (
            resumes['raw']
            .str.replace('. ', ' ')  # Replace periods with spaces
            .str.replace('\n', ' ')  # Replace newlines with spaces
            .apply(self.match_skills_by_spacy)
        )
        return resumes
