from spacy.lang.en import English


class JobInfoExtraction:

    def __init__(self, skills_patterns_path):
        self.skills_patterns_path = skills_patterns_path
        self.nlp = English()
        ruler = self.nlp.add_pipe("entity_ruler")
        ruler.from_disk(self.skills_patterns_path)

    def match_skills_by_spacy(self, job, display=False):
        doc1 = self.nlp(job)
        job_skills = []
        for ent in doc1.ents:
            labels_parts = ent.label_.split('|')
            if labels_parts[0] == 'SKILL':
                if display: print((ent.text, ent.label_))
                if labels_parts[1].replace('-', ' ') not in job_skills:
                    job_skills.append(labels_parts[1].replace('-', ' '))
        return job_skills

    def extract_entities(self, jobs):
        jobs['Skills'] = jobs['raw'].str.replace('.', ' ').str.replace('\n', ' ').apply(self.match_skills_by_spacy)
        return jobs