from spacy.lang.en import English

class ResumeInfoExtraction:

    def __init__(self, skills_patterns_path, name):
        self.name = name
        self.skills_patterns_path = skills_patterns_path
        self.nlp = English()
        ruler = self.nlp.add_pipe("entity_ruler")
        ruler.from_disk(self.skills_patterns_path)

    def match_skills_by_spacy(self, resume, display=False):
        doc1 = self.nlp(resume)
        resume_skills = []
        for ent in doc1.ents:
            labels_parts = ent.label_.split('|')
            if labels_parts[0] == 'SKILL':
                if display: print((ent.text, ent.label_))
                if labels_parts[1].replace('-', ' ') not in resume_skills:
                    resume_skills.append(labels_parts[1].replace('-', ' '))
        return resume_skills

    def extract_entities(self, resumes):
        resumes['Skills'] = resumes['raw'].str.replace('. ', ' ').str.replace('\n', ' ').apply(self.match_skills_by_spacy)
        return resumes