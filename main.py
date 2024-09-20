import os
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
import pandas as pd
import json
from services.ResumeInfoExtraction import ResumeInfoExtraction
from services.JobInfoExtraction import JobInfoExtraction
from source.schemas.resumeextracted import ResumeExtractedModel
from source.schemas.jobextracted import JobExtractedModel
import ast
from pypdf import PdfReader
import warnings 
import logging
from cryptography.utils import CryptographyDeprecationWarning

logging.getLogger('pypdf').setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)



def get_resumes(directory):
    
    def extract_pdf(path):
        reader = PdfReader(path)
        number_of_pages = len(reader.pages)
        text = ""
        for i in range(number_of_pages):
            page = reader.pages[i]
            text += page.extract_text()
        return text
    
    dic = {}
    
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        if os.path.isfile(file_path) and filename.endswith(".pdf"):
            name = filename.strip(".pdf")
            resume_text = extract_pdf(file_path)
            dic[name] = [resume_text]
    
    df = pd.DataFrame(dic).T
    df.reset_index(inplace=True)
    df.rename(columns={"index": "name", 0:"raw"}, inplace=True)
    
    return df

def transform_dataframe_to_json(dataframe):

    # transforms the dataframe into json
    result = dataframe.to_json(orient="records")
    parsed = json.loads(result)
    json_data = json.dumps(parsed, indent=4)

    return json_data


def resume_extraction(resume):
    degrees_patterns_path = 'Resources/data/degrees.jsonl'
    majors_patterns_path = 'Resources/data/majors.jsonl'
    skills_patterns_path = 'Resources/data/skills.jsonl'
    jobs = resume
    names = transform_dataframe_to_json(jobs[["name"]])
    job_extraction = ResumeInfoExtraction(skills_patterns_path, majors_patterns_path, degrees_patterns_path, jobs, names)
    jobs = job_extraction.extract_entities(jobs)
    for i, row in jobs.iterrows():
        name = row["name"]
        degrees = jobs.loc[i, 'Degrees']
        maximum_degree_level = jobs.loc[i, 'Maximum degree level']
        acceptable_majors = jobs.loc[i, 'Acceptable majors']
        skills = jobs.loc[i, 'Skills']
        

        job_extracted = ResumeExtractedModel(maximum_degree_level=maximum_degree_level if maximum_degree_level else '',
                                          acceptable_majors=acceptable_majors if acceptable_majors else [],
                                          skills=skills if skills else [],
                                          name=name if name else '',
                                          degrees=degrees if degrees else [])
        job_extracted = jsonable_encoder(job_extracted)
    jobs_json = transform_dataframe_to_json(jobs)
    
    return jobs_json


def job_info_extraction(resume):
    degrees_patterns_path = 'Resources/data/degrees.jsonl'
    majors_patterns_path = 'Resources/data/majors.jsonl'
    skills_patterns_path = 'Resources/data/skills.jsonl'
    jobs = resume
    job_extraction = JobInfoExtraction(skills_patterns_path, majors_patterns_path, degrees_patterns_path, jobs)
    jobs = job_extraction.extract_entities(jobs)
    for i, row in jobs.iterrows():
        minimum_degree_level = jobs['Minimum degree level'][i]
        acceptable_majors = jobs['Acceptable majors'][i]
        skills = jobs['Skills'][i]

        job_extracted = JobExtractedModel(minimum_degree_level=minimum_degree_level if minimum_degree_level else '',
                                          acceptable_majors=acceptable_majors if acceptable_majors else [],
                                          skills=skills if skills else [])
        job_extracted = jsonable_encoder(job_extracted)
        # new_job_extracted = database.get_collection("jobsextracted").insert_one(job_extracted)
    jobs_json = transform_dataframe_to_json(jobs)
    return jobs_json

if __name__ == "__main__":
    # Create DF for resumes
    df = get_resumes("resumes")
    res = resume_extraction(df)
    df = pd.read_json(res)
    print(df)

    # Create DF for jobs
    with open('job_descriptions/description.txt', 'r') as file:
        job_description = file.read()

    job_description = [job_description]
    df2 = pd.DataFrame(job_description, columns=["raw"])
    res = job_info_extraction(df2)
    df2 = pd.read_json(res)
    print(df2)
    