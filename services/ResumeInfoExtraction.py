from Resources import DEGREES_IMPORTANCE
from spacy.lang.en import English

class ResumeInfoExtraction:

    def __init__(self, skills_patterns_path, majors_patterns_path, degrees_patterns_path, jobs, name):
        self.name = name
        self.skills_patterns_path = skills_patterns_path
        self.majors_patterns_path = majors_patterns_path
        self.degrees_patterns_path = degrees_patterns_path
        self.degrees_importance = DEGREES_IMPORTANCE

    @staticmethod
    def match_majors_by_spacy(self, job):
        nlp = English()
        # Add the pattern to the matcher
        patterns_path = self.majors_patterns_path
        ruler = nlp.add_pipe("entity_ruler")
        ruler.from_disk(patterns_path)
        # Process some text
        doc1 = nlp(job)
        acceptable_majors = []
        for ent in doc1.ents:
            labels_parts = ent.label_.split('|')
            if labels_parts[0] == 'MAJOR':
                if labels_parts[2].replace('-', ' ') not in acceptable_majors:
                    acceptable_majors.append(labels_parts[2].replace('-', ' '))
                if labels_parts[2].replace('-', ' ') not in acceptable_majors:
                    acceptable_majors.append(labels_parts[2].replace('-', ' '))
        return acceptable_majors

    @staticmethod
    def match_degrees_by_spacy(self, job, display=False):
        nlp = English()
        # Add the pattern to the matcher
        patterns_path = self.degrees_patterns_path
        ruler = nlp.add_pipe("entity_ruler")
        ruler.from_disk(patterns_path)
        # Process some text
        doc1 = nlp(job)
        degree_levels = []
        for ent in doc1.ents:
            labels_parts = ent.label_.split('|')
            if labels_parts[0] == 'DEGREE':
                if display: print((ent.text, ent.label_))
                if labels_parts[1] not in degree_levels:
                    degree_levels.append(labels_parts[1])
        return degree_levels

    @staticmethod
    def match_skills_by_spacy(self, job, display=False):
        nlp = English()
        patterns_path = self.skills_patterns_path
        ruler = nlp.add_pipe("entity_ruler")
        ruler.from_disk(patterns_path)
        # Process some text
        doc1 = nlp(job)
        job_skills = []
        for ent in doc1.ents:
            labels_parts = ent.label_.split('|')
            if labels_parts[0] == 'SKILL':
                if display: print((ent.text, ent.label_))
                if labels_parts[1].replace('-', ' ') not in job_skills:
                    job_skills.append(labels_parts[1].replace('-', ' '))
        return job_skills

    @staticmethod
    def get_maximum_degree(self, degrees):
        """get the Maximum degree that the candidate has"""
        d = {degree: self.degrees_importance[degree] for degree in degrees}
        return max(d, key=d.get)

    def extract_entities(self, jobs):
        # Initialize columns
        jobs['Degrees'] = ""
        jobs['Maximum degree level'] = ""
        jobs['Acceptable majors'] = ""
        jobs['Skills'] = ""
        
        for i, row in jobs.iterrows():
            job = jobs['raw'][i].replace('. ', ' ')
            degrees = self.match_degrees_by_spacy(self, job)
            if len(degrees) != 0:
                jobs['Maximum degree level'][i] = self.get_maximum_degree(self, degrees)
            else:
                jobs['Maximum degree level'][i] = ""
            jobs['Degrees'][i] = degrees
            jobs['Acceptable majors'][i] = self.match_majors_by_spacy(self, job)
            jobs['Skills'][i] = self.match_skills_by_spacy(self, job)

        # Reorder columns
        column_order = ['name', 'raw', 'Degrees', 'Maximum degree level', 'Acceptable majors', 'Skills']
        jobs = jobs[column_order]
        
        return jobs