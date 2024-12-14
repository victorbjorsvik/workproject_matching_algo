import os
import pandas as pd
import numpy as np
from extraction.ResumeInfoExtraction import ResumeInfoExtraction
from extraction.JobInfoExtraction import JobInfoExtraction
from source.schemas.resumeextracted import ResumeExtractedModel # Let's reintroduce later on
from source.schemas.jobextracted import JobExtractedModel # Let's reintroduce later on
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
import markdown
import warnings 
import logging
import torch
import torch.nn.functional as F

# Handle annoying warnings
logging.getLogger('pypdf').setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# Get the absolute path of the root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Paths to your pattern files
skills_patterns_path = os.path.join(ROOT_DIR, 'workproject_matching_algo','patterns', 'skills.jsonl')
skills_patterns_path_2 = os.path.join(ROOT_DIR, 'workproject_matching_algo','patterns', 'skills_branch.jsonl')


def get_resumes(directory):
    """ Function to parse and extract text from PDFs in a directory """
    
    def extract_pdf(path):
        """ Helper function to extract the text from the PDFs using the PyMuPDF library"""
        try:
            # Use PyMuPDF to parse and extract text from resumes
            with fitz.open(path) as doc:
                text = ''.join(page.get_text() for page in doc)
            return text
        except Exception as e:
            logging.error(f"Error processing {path}: {e}")
            return ""

    # initialize empty dictionary
    dic = {}
    
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        # Extract text from pdf if file is pdf
        if os.path.isfile(file_path) and filename.endswith(".pdf"):
            name = filename.rstrip(".pdf")
            resume_text = extract_pdf(file_path)
            dic[name] = [resume_text]
    
    # Create a pandas dataframe from the parsed PDFs
    df = pd.DataFrame(dic).T
    df.reset_index(inplace=True)
    df.rename(columns={"index": "name", 0: "raw"}, inplace=True)
    
    return df


def resume_extraction(resumes):
    """ function to extract the relevant skills from a resume """
    names = resumes[["name"]]
    resume_extraction = ResumeInfoExtraction(skills_patterns_path_2, names)
    resumes_df = resume_extraction.extract_entities(resumes)
    return resumes_df


def job_info_extraction(jobs):
    """ function to extract the relevant skills from a job description """
    job_extraction = JobInfoExtraction(skills_patterns_path)
    job_df = job_extraction.extract_entities(jobs)
    return job_df


def calc_similarity_baseline(applicant_df, job_df, N=3, parallel=False):
    """Calculate cosine similarity based on MPNET embeddings of combined skills."""

    # Initialize the model
    model = SentenceTransformer('all-mpnet-base-v2')
    model.max_seq_length = 75
    model.tokenizer.padding_side="right"
    model.eval()

    def add_eos(input_examples):
        """ helper function to add special tokens between each skills"""
        input_examples = [input_example + model.tokenizer.eos_token for input_example in input_examples]
        return input_examples

    # Precompute job embeddings
    job_df['Skills_Text'] = job_df['Skills'].apply(add_eos) # add special tokens
    job_df['Skills_Text'] = job_df['Skills_Text'].apply(lambda x: ' '.join(sorted(set(x))) if isinstance(x, list) else '') # concatenate into one string
    job_embeddings = model.encode(job_df['Skills_Text'].tolist()) # encode job descriptions
    # Precompute applicant embeddings
    applicant_df['Skills_Text'] = applicant_df['Skills'].apply(add_eos) # add special tokens
    applicant_df['Skills_Text'] = applicant_df['Skills_Text'].apply(lambda x: ' '.join(sorted(set(x))) if isinstance(x, list) else '') # concatenate into one string
    applicant_embeddings = model.encode(
        applicant_df['Skills_Text'].tolist(),
        batch_size=32,
        num_workers=os.cpu_count() // 2 if parallel else 0, # parallelize
        show_progress_bar=False
    )

    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(job_embeddings, applicant_embeddings)

    # Create a DataFrame from the similarity matrix
    similarity_df = pd.DataFrame(similarity_matrix.T, index=applicant_df['name'], columns=job_df.index)
    similarity_df = similarity_df.reset_index().melt(id_vars='name', var_name='job_id', value_name='similarity_score')
    similarity_df['rank'] = similarity_df.groupby('job_id')['similarity_score'].rank(ascending=False)
    similarity_df['interview_status'] = similarity_df['rank'].apply(lambda x: 'Selected' if x <= N else 'Not Selected')

    return similarity_df.sort_values("rank")


def calc_cross(applicant_df, job_df, N=3, parallel=False):
    """ Use Cross Encoder to calculate similarity of combined skills."""

    # Initialize the model once outside the loop for efficiency
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

     # Precompute job embeddings
    job_df['Skills_Text'] = job_df['Skills'].apply(lambda x: ' '.join(sorted(set(x))) if isinstance(x, list) else '')
    query = job_df['Skills_Text'][0]
    # Precompute applicant embeddings
    applicant_df['Skills_Text'] = applicant_df['Skills'].apply(lambda x: ' '.join(sorted(set(x))) if isinstance(x, list) else '')
    applicants = applicant_df['Skills_Text'].tolist()

    ranks = model.rank(
        query,
        applicants,
        batch_size=32,
        num_workers=os.cpu_count() // 2 if parallel else 0,
        show_progress_bar=False
    )

    similarity_df = pd.DataFrame(ranks)
    similarity_df['softmaxed'] = F.softmax(torch.tensor(similarity_df['score']))
    similarity_df = similarity_df.join(applicant_df[["name"]], on="corpus_id")
    
    # similarity_df['interview_status'] = similarity_df.index.apply(lambda x: 'Selected' if x <= N else 'Not Selected')

    return similarity_df

def calc_similarity(applicant_df, job_df, N=3, parallel=False):
    """Calculate cosine similarity based on BERT embeddings of skills (skill-by-skill)."""

    def semantic_similarity_all_MiniLM_L6_v2(job, resume):
        """Calculate similarity with all-MiniLM-L6-v2."""
        model = SentenceTransformer('all-MiniLM-L6-v2')
        model.eval()
        score = 0
        sen = job + resume
        sen_embeddings = model.encode(sen,
                                      batch_size=4,
                                      device='cpu',
                                      show_progress_bar=False,
                                      num_workers=os.cpu_count() // 2 if parallel else 0
                                      )
        for i in range(len(job)):
            if job[i] in resume:
                score += 1
            else:
                max_cosine_sim = max(cosine_similarity([sen_embeddings[i]], sen_embeddings[len(job):])[0])
                if max_cosine_sim >= 0.4:
                    score += max_cosine_sim
        score = score / len(job)
        return round(score, 3)

    # Prepare a DataFrame to store results
    matching_dataframe = []

    # Loop through each job in the job_df
    for job_index in range(len(job_df)):
        job_skills = job_df['Skills'].iloc[job_index]  # Use iloc for positional indexing

        # Loop through each applicant in the applicant_df
        for applicant_id in range(len(applicant_df)):
            applicant_skills = applicant_df['Skills'].iloc[applicant_id]  # Use iloc for positional indexing
            applicant_name = applicant_df['name'].iloc[applicant_id]  # Ensure correct column access

            # Compute similarity score
            score = semantic_similarity_all_MiniLM_L6_v2(job_skills, applicant_skills)

            # Append result to the DataFrame
            matching_dataframe.append({
                "name": applicant_name,
                "job_id": job_index,
                "similarity_score": score
            })

    # Create a DataFrame from results
    matching_dataframe = pd.DataFrame(matching_dataframe)

    # Add rank based on similarity score
    matching_dataframe['rank'] = matching_dataframe['similarity_score'].rank(ascending=False)
    matching_dataframe['interview_status'] = matching_dataframe['rank'].apply(lambda x: 'Selected' if x <= N else 'Not Selected')
    
    return matching_dataframe.sort_values("rank")


def role_similarity(cv_df, job_df, hard_skills, scores_df, titles_df, growth_df, role=None, wage=0 , parallel=False):
    """
    Calculate cosine similarity between a single CV and multiple job descriptions.
    Optionally filter based on a specific role and include role-specific scores in the output.
    """
    # Initialize the model
    model = SentenceTransformer('all-mpnet-base-v2')
    model.max_seq_length = 75
    model.tokenizer.padding_side = "right"
    model.eval()

    def add_eos(input_examples):
        """Helper function to add special tokens between each skill."""
        input_examples = [input_example + model.tokenizer.eos_token for input_example in input_examples]
        return input_examples

    def missing_skills(skills, cv):
        """Return a list of words from the job description (title and skills) that do not match any keywords in the CV."""
        skill_words = set(skill.lower().strip() for skill in skills if isinstance(skill, str))
        cv_set = set(cv.lower().strip() for cv in cv if isinstance(cv, str))
        missing_words = skill_words - cv_set
        return list(missing_words)

    # Find the generic title (Role) corresponding to the input title (Reported Job Title)
    generic_title = titles_df.loc[titles_df['Reported Job Title'] == role, 'Title'].iloc[0]
    
    # Filter scores_df using the generic title
    filtered_scores_df = scores_df[scores_df['Target_Role'] == generic_title]
    # Map the filtered scores to a dictionary for quick lookup
    role_score_mapping = dict(zip(filtered_scores_df['Role'], filtered_scores_df['composite_tasks_dwas_ksas']))


    # Extract and process the CV's skills
    cv_df['Skills_Text'] = cv_df['Skills'].apply(add_eos)
    cv_df['Skills_Text'] = cv_df['Skills_Text'].apply(
        lambda x: ' '.join(sorted(set(x))) if isinstance(x, list) else ''
    )
    cv_embedding = model.encode(
        cv_df['Skills_Text'].iloc[0],  # Assuming a single CV is provided
        batch_size=1,
        show_progress_bar=False
    )

    # Get job embeddings from DataFrame
    job_embeddings = np.vstack(job_df['skills_embed'].values)

    # Compute cosine similarity
    similarity_scores = cosine_similarity([cv_embedding], job_embeddings).flatten()
    job_df['similarity_score'] = similarity_scores

    # Normalize similarity scores using z-score
    mean_score = job_df['similarity_score'].mean()
    std_score = job_df['similarity_score'].std()
    if std_score > 0:
        job_df['normalized_similarity_score'] = (job_df['similarity_score'] - mean_score) / std_score
    else:
        job_df['normalized_similarity_score'] = 0  # Assign 0 if std is zero (all scores are identical)

    # Rank the jobs by normalized similarity score
    job_df['rank'] = job_df['normalized_similarity_score'].rank(ascending=False)

    # Add missing skills
    job_df['missing_skills'] = job_df.apply(
        lambda row: missing_skills(
            hard_skills.loc[row.name, 'Skills'],  # Skills from df_skill_role_grouped
            cv_df['Skills'].iloc[0]  # Skills from the first row of df_resumes
        ),
        axis=1
    )

    # Map the growth information using generic title
    role_growth_mapping = dict(zip(growth_df['Occupation'], growth_df['Categories']))
    job_df['role_growth'] = job_df['Title'].map(role_growth_mapping)

    # Fill NaN values in 'role_growth' with 'Not In-Demand'
    job_df['role_growth'].fillna('Not In-Demand', inplace=True)

    # Check for missing salary information
    job_df['annual_wage_variation'] = job_df.apply(
        lambda row: "Info Not Available" if row['annual_wage'] == 0 else row['annual_wage'] - wage,
        axis=1
    )

    job_df_2 = job_df.copy()
    # Add the role-specific scores to a new column
    job_df_2['role_scores'] = job_df_2['Title'].apply(lambda x: role_score_mapping.get(x, None))
    # Rank the jobs by normalized similarity score
    job_df_2['rank'] = job_df_2['role_scores'].rank(ascending=False)

    # Return a DataFrame with the job titles, similarity scores, normalized scores, role-specific scores, and ranks
    ranked_jobs_2 = job_df_2[['rank', 'Title', 'missing_skills', 'annual_wage_variation', 'role_growth']].sort_values(by='rank', ascending=True)
    ranked_jobs = job_df[['rank', 'Title', 'missing_skills', 'annual_wage_variation', 'role_growth']].sort_values(by='rank', ascending=True)
    return ranked_jobs, ranked_jobs_2


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def tailored_questions(api_key, applicants, required_skills, model="gpt-4o-mini"):
    """
    Function to create tailored interview questions using OpenAI's API.

    :param api_key: str, the OpenAI API key for authentication.
    :param applicants: str, a list of candidates' names and their skills.
    :param required_skills: str, the skills required for the job.
    :param model: str, the OpenAI model to use (default: "gpt-4o-mini").
    :return: str, HTML content of the tailored interview questions.
    """
    prompt = f"""
    We have a list of candidates and wish to generate tailored interview questions based on their skills and the skills required for the job. Note that all candidates should receive the same set of questions to ensure fairness and avoid bias.

    Please generate a list of **5 self-contained interview questions** that:
    - Assess the overlap between the candidates' skills and the required job skills.
    - Are relevant to the job's responsibilities and the provided skillsets.
    - Cover both technical skills and problem-solving abilities, where applicable.
    - Are phrased clearly and concisely.

    Here is the information:
    - **Candidates and their skills**: {applicants}
    - **Required job skills**: {required_skills}

    The output should follow this structure:
    1. **Question 1**: [Write the first question.]
    2. **Question 2**: [Write the second question.]
    3. **Question 3**: [Write the third question.]
    4. **Question 4**: [Write the fourth question.]
    5. **Question 5**: [Write the fifth question.]

    Each question should challenge the candidates to demonstrate their proficiency in the relevant areas while being answerable based on their stated skills.
    """

    # Call OpenAI's API to generate the response
    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful recruiting assistant. Your job is to generate fair, skill-based interview questions tailored to a set of candidates' skills and the job's requirements."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1000
    )

    markdown_output = completion.choices[0].message.content
    html_output = markdown.markdown(markdown_output)  # Convert markdown to HTML

    return html_output



@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def generate_coding_exercise(api_key, job_description, model="gpt-4"):
    """
    Generate a tailored coding exercise based on a job description using OpenAI's API.

    :param job_description: str, the job description to base the exercise on
    :return: str, the coding exercise prompt
    """
    prompt = f"""
    Based on the following job description, create a self-contained practical coding exercise for a coding interview. The exercise should:
    - Be solvable using Python.
    - Be tailored to the job description's requirements and responsibilities.
    - Include clear instructions for the candidate.
    - Require no external dataset and be solvable based on the prompt alone.
    - Follow this structure:

    1. **Scenario**: Provide a realistic scenario relevant to the job.
    2. **Instructions**: Step-by-step guidance on what the candidate needs to do.
    3. **Expected Deliverables**: Clearly define what is expected from the candidate.
    4. **Sample Code Structure**: Offer a template or skeleton code for the candidate to follow.
    5. **Assessment Criteria**: Outline how the solution will be evaluated.

    Job Description:
    {job_description}

    Generate the exercise now.
    """

    # Call OpenAI's API to generate the response
    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates coding exercises for job interviews."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1000
    )

    markdown_output = completion.choices[0].message.content
    return markdown_output


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def bespoke_apologies(api_key, applicants, required_skills, model="gpt-4o-mini"):
    """
    Function to create bespoke apology letters using OpenAI's API.

    :param api_key: str, the OpenAI API key for authentication.
    :param applicants: str, a list of candidates' names and their skills.
    :param required_skills: str, the skills required for the job.
    :param model: str, the OpenAI model to use (default: "gpt-4o-mini").
    :return: str, HTML content of the bespoke apology letters.
    """
    prompt = f"""
    We have a list of candidates for a job, but unfortunately, none of them made it to the first round of interviews. 
    Please generate a **bespoke apology letter** for each candidate that:

    - Addresses the candidate by name.
    - Explains respectfully that their skills were not a perfect match for the job's requirements.
    - Offers encouragement and constructive feedback.
    - Provides specific resources or advice to help them improve their skills for future opportunities.

    Here is the information:
    - **Candidates and their skills**: {applicants}
    - **Required job skills**: {required_skills}

    Each apology letter should:
    - Start with a polite and empathetic opening.
    - Acknowledge the candidate's effort in applying for the role.
    - Highlight their strengths, if applicable.
    - Suggest practical resources (e.g., courses, books, online materials) tailored to the skills they are lacking.
      Provide links to relevant online courses and list them like 1., 2. and 3. on seperate lines.
    - End on an encouraging note, wishing them success in their career.

    Output the letters in a structured format:
    - **Candidate Name**: [Write the bespoke apology letter here.]

    Generate the bespoke apologies now.
    """

    # Call OpenAI's API to generate the response
    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful and empathetic recruiting assistant. "
                    "Your task is to create polite, constructive, and encouraging apology letters "
                    "for candidates who did not proceed to the first round of interviews."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7,
        max_tokens=1000
    )

    markdown_output = completion.choices[0].message.content
    html_output = markdown.markdown(markdown_output)  # Convert markdown to HTML

    return html_output



######################################################################################
###                                  SCRIPT                                        ###
######################################################################################
import time # for benchmarking during code optimization


def main(open_ai=False):
    t0 = time.time()
    # Create DataFrame for resumes
    df_resumes = get_resumes("resumes")
    df_resumes = resume_extraction(df_resumes)
    if not open_ai:
        print(df_resumes[["name", "Skills"]])

    # Create DataFrame for jobs
    description_file_path = os.path.join(ROOT_DIR, 'workproject_matching_algo', 'job_descriptions', 'job2.txt')
    with open(description_file_path, 'r') as file:
        job_description = file.read()
    df_jobs = pd.DataFrame([job_description], columns=["raw"])
    df_jobs = job_info_extraction(df_jobs)
    if not open_ai:
        print(df_jobs)

    # Conduct Similarity Analysis
    analysis_data = calc_similarity(df_resumes, df_jobs, parallel=True)
    # analysis_data_df = calc_cross(df_resumes, df_jobs, parallel=True)
    if not open_ai:
        print(analysis_data.sort_values("rank"))
        # print(analysis_data_df)

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
    
