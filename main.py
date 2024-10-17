import os
import pandas as pd
import json
from services.ResumeInfoExtraction import ResumeInfoExtraction
from services.JobInfoExtraction import JobInfoExtraction
from source.schemas.resumeextracted import ResumeExtractedModel # Let's reintroduce later on
from source.schemas.jobextracted import JobExtractedModel # Let's reintroduce later on
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
#import openai
import json
import warnings 
import logging
import os
import json
import pandas as pd
logging.getLogger('pypdf').setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# Get the absolute path of the root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Paths to your pattern files
degrees_patterns_path = os.path.join(ROOT_DIR, 'workproject_matching_algo', 'Resources', 'data', 'degrees.jsonl')
majors_patterns_path = os.path.join(ROOT_DIR, 'workproject_matching_algo', 'Resources', 'data', 'majors.jsonl')
skills_patterns_path = os.path.join(ROOT_DIR, 'workproject_matching_algo','Resources', 'data', 'skills.jsonl')



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


def resume_extraction(resumes):
    resumes = resumes.copy()
    names = resumes[["name"]]
    resume_extraction = ResumeInfoExtraction(skills_patterns_path, majors_patterns_path, degrees_patterns_path, resumes, names)
    resumes_df = resume_extraction.extract_entities(resumes)
    return resumes_df


def job_info_extraction(jobs):
    jobs = jobs.copy()
    job_extraction = JobInfoExtraction(skills_patterns_path, majors_patterns_path, degrees_patterns_path, jobs)
    job_df = job_extraction.extract_entities(jobs)
    return job_df


def calc_similarity(applicant_df, job_df, N=3):
    """Calculate cosine similarity based on BERT embeddings of combined skills."""

    # Initialize the model once outside the loop for efficiency
    model = SentenceTransformer('all-mpnet-base-v2')
    model.eval()

    columns = ['applicant', 'job_id', 'all-mpnet-base-v2_score']
    matching_dataframe = pd.DataFrame(columns=columns)

    for job_index in range(job_df.shape[0]):
        # Get the list of skills for the current job
        job_skills = job_df['Skills'][job_index]
        # Remove duplicates and sort the skills
        job_skills = sorted(set(job_skills)) if isinstance(job_skills, list) else []
        # Combine job skills into a single string
        job_text = ' '.join(job_skills)

        matching_data = []

        for applicant_id in range(applicant_df.shape[0]):
            # Get applicant name and skills
            applicant_name = applicant_df.iloc[applicant_id]['name']
            applicant_skills = applicant_df['Skills'][applicant_id]
            # Remove duplicates and sort the skills
            applicant_skills = sorted(set(applicant_skills)) if isinstance(applicant_skills, list) else []
            # Combine applicant skills into a single string
            resume_text = ' '.join(applicant_skills)

            # Encode the combined skills for job and applicant
            embeddings = model.encode([job_text, resume_text])

            # Compute cosine similarity between the job and applicant embeddings
            score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

            matching_dataframe_job = {
                "applicant": applicant_name,
                "job_id": job_index,
                "all-mpnet-base-v2_score": round(score, 3)
            }
            matching_data.append(matching_dataframe_job)

        # Append the matching data to the dataframe
        matching_dataframe = pd.concat([matching_dataframe, pd.DataFrame(matching_data)], ignore_index=True)

    # Rank the applicants based on their similarity scores
    matching_dataframe['rank'] = matching_dataframe['all-mpnet-base-v2_score'].rank(ascending=False)
    matching_dataframe['interview_status'] = matching_dataframe['rank'].apply(lambda x: 'Selected' if x <= N else 'Not Selected')

    return matching_dataframe

def bespoke_apologies():
    return None


if __name__ == "__main__":
    # Create DataFrame for resumes
    df_resumes = get_resumes("resumes")
    df_resumes = resume_extraction(df_resumes)
    print(df_resumes[["name", "Skills"]])

    # Create DataFrame for jobs
    description_file_path = os.path.join(ROOT_DIR, 'workproject_matching_algo', 'job_descriptions', 'job1.txt')
    with open(description_file_path, 'r') as file:
        job_description = file.read()

    df_jobs = pd.DataFrame([job_description], columns=["raw"])
    df_jobs = job_info_extraction(df_jobs)
    # print(df_jobs)

    analysis_data_df = calc_similarity(df_resumes, df_jobs)
    print(analysis_data_df.sort_values("rank", ascending=True ))

    