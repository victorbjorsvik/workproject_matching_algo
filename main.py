import os
import pandas as pd
from services.ResumeInfoExtraction import ResumeInfoExtraction
from services.JobInfoExtraction import JobInfoExtraction
from source.schemas.resumeextracted import ResumeExtractedModel # Let's reintroduce later on
from source.schemas.jobextracted import JobExtractedModel # Let's reintroduce later on
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
import markdown
import warnings 
import logging
logging.getLogger('pypdf').setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# Get the absolute path of the root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Paths to your pattern files
skills_patterns_path = os.path.join(ROOT_DIR, 'workproject_matching_algo','Resources', 'data', 'skills.jsonl')


def get_resumes(directory):
    
    def extract_pdf(path):
        try:
            with fitz.open(path) as doc:
                text = ''.join(page.get_text() for page in doc)
            return text
        except Exception as e:
            logging.error(f"Error processing {path}: {e}")
            return ""

    
    dic = {}
    
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        if os.path.isfile(file_path) and filename.endswith(".pdf"):
            name = filename.rstrip(".pdf")
            resume_text = extract_pdf(file_path)
            dic[name] = [resume_text]
    
    df = pd.DataFrame(dic).T
    df.reset_index(inplace=True)
    df.rename(columns={"index": "name", 0: "raw"}, inplace=True)
    
    return df


def resume_extraction(resumes):
    names = resumes[["name"]]
    resume_extraction = ResumeInfoExtraction(skills_patterns_path, names)
    resumes_df = resume_extraction.extract_entities(resumes)
    return resumes_df


def job_info_extraction(jobs):
    job_extraction = JobInfoExtraction(skills_patterns_path)
    job_df = job_extraction.extract_entities(jobs)
    return job_df


def calc_similarity(applicant_df, job_df, N=3, parallel=False):
    """Calculate cosine similarity based on BERT embeddings of combined skills."""

    # Initialize the model once outside the loop for efficiency
    model = SentenceTransformer('all-mpnet-base-v2')
    model.eval()

     # Precompute job embeddings
    job_df['Skills_Text'] = job_df['Skills'].apply(lambda x: ' '.join(sorted(set(x))) if isinstance(x, list) else '')
    job_embeddings = model.encode(job_df['Skills_Text'].tolist())
    # Precompute applicant embeddings
    applicant_df['Skills_Text'] = applicant_df['Skills'].apply(lambda x: ' '.join(sorted(set(x))) if isinstance(x, list) else '')
    import code; code.interact(local=locals()) 
    applicant_embeddings = model.encode(
        applicant_df['Skills_Text'].tolist(),
        batch_size=32,
        num_workers=os.cpu_count() // 2 if parallel else 0,
        show_progress_bar=False
    )

    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(job_embeddings, applicant_embeddings)

    # Create a DataFrame from the similarity matrix
    similarity_df = pd.DataFrame(similarity_matrix.T, index=applicant_df['name'], columns=job_df.index)
    similarity_df = similarity_df.reset_index().melt(id_vars='name', var_name='job_id', value_name='similarity_score')
    similarity_df['rank'] = similarity_df.groupby('job_id')['similarity_score'].rank(ascending=False)
    similarity_df['interview_status'] = similarity_df['rank'].apply(lambda x: 'Selected' if x <= N else 'Not Selected')

    return similarity_df

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def tailored_questions(api_key,  applicants, required_skills, model="gpt-4o-mini"):
    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a helpful recruiting assistant. We have a list of candidates we want to interview for a job and we want to tailor interview questions to their skills. Note: to avoid bias we want the applicants to recieive the same questions"}, # <-- This is the system message that provides context to the model
        {"role": "user", "content": f"Hello! Based on the following candidates: {applicants}, could you make a list of 5 interview questions for all of them based on their total pool of skills and how it relates to the skills required of the job - here: {required_skills} "}  # <-- This is the user message for which the model will generate a response
    ]
    )

    markdown_output = completion.choices[0].message.content
    html_output = markdown.markdown(markdown_output)  # Convert markdown to HTML

    return html_output

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def bespoke_apologies(api_key,  applicants, required_skills, model="gpt-4o-mini"):
    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a helpful recruiting assistant. We have a list of candidates for a job, but unfortunately none of them made it to the first round of interviews."}, # <-- This is the system message that provides context to the model
        {"role": "user", "content": f"""Hello! Based on the following candidates: {applicants}, could you make a bespoke aplogy letter to each of them and explain that their skills were not a 
        prefect match with the required skills here:{required_skills}. For each of the applicants, please also provide them with some resources to improve the skills in which they are lacking so they have better chances in the next round of recruiting """}  # <-- This is the user message for which the model will generate a response
    ]
    )

    markdown_output = completion.choices[0].message.content
    html_output = markdown.markdown(markdown_output)  # Convert markdown to HTML

    return html_output


######################################################################################
###                                  SCRIPT                                        ###
######################################################################################
import time


def main(open_ai=False):
    t0 = time.time()
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
    print(df_jobs)

    # Conduct Similarity Analysis
    analysis_data_df = calc_similarity(df_resumes, df_jobs, parallel=True)
    print(analysis_data_df.sort_values("rank", ascending=True ))

    t1 = time.time()
    dt = t1 - t0
    print(f"dt: {dt*1000:.2f}ms")

    if open_ai:    
        # Set the API key and model name
        MODEL="gpt-4o-mini"
        api_key=os.getenv("OPENAI_API_KEY", "<your OpenAI API key if not set as an env var>")

        # Create tailored interview questions
        tailored_questions = tailored_questions(api_key, df_resumes, df_jobs['Skills'], model=MODEL)
        print(tailored_questions)

        # Create bespoke apologies
        bespoke_apologies = bespoke_apologies(api_key, df_resumes, df_jobs['Skills'], model=MODEL)
        print(bespoke_apologies)


if __name__ == "__main__":
    main(open_ai=False)
    
