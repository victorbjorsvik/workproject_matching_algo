# AGENDA:

1. Walk you through the whole code base (GOAL: know what every directory does)
2. Walk you through the main module (GOAL: understand how the logic of the matching algo looks is translated to code)
3. (Optional) Walk you through the user_interface directory (GOAL: understand how we can implement our model as logic in the backend of a web application)



Front end:

    * HTML
    * CSS 
    * JavaSCript

Back end:

    * Flask 



## CHALLANGES:
 * Data: We have no trainabale parameters per se, so data won't help us improve the model (could help us evaluate though*)
 * Data: Could help us evaluate, but unless company used skill-based hiring we are measuring our model's performance based on traditional hiring methods heuristics 
    (in our part 1 we argue that skills-first is "superior" to traditional hiring - makes little sense to use traditional hiring targets to evaluate our model)
 * Data: Even if we were to obtain the "perfect dataset" (e.g. job descriptions, applicant resumes and lables), our hypothesis is still that a skills-first approach to hiring
    could lead to better fit in empolyment than traditional methods. 1) By training our model (introducing more trainable parameters), on this data we would skew its weights to conform
    with the traditional hiring methods. 2) By evaluating our model on this dataset we would compare what we believe is the "superior" method to the "less ideal method". Moreover, there is 
    a challange with regards to which labels we should look for/ choose: {proceed_to_interview(0,1), got_the_job(0,1), productivity_after_employed(continous)}. To prove our hypothesis
    and the efficacy of our model we would need the latter.
    ** Skills first does not imply that the applicant is more likely to get hired during traditional rounds of interviews, rather it argues that the pool pf relevant applicants will be larger
    and that productivity and employye match might be better** 


 * To prove that our model is better than traditional approaches we would neeed to gather productivity data on the people hired based on our model and measure
    their productivity against productivity of people hired via traditional methods. 
 


 When making our matching model we run into the "coldstart" problem with our recommender system. An idea we had was to train a transformer on data, but due to a lack of



## TODO:
  * **(PARADIGM)** Rephrase our thinking: We are actually providing new value based on the skills-first approach. Not creating the best possible matching algo based on traditional heurisitcs. Argument derived from EDA and LR.
  * **(SYNTHETIC DATA & EVALUATION)** Still need some way to evalute models - synthetic data is probably the way to go:
         * Let's gather some job descriptions or CVs (REAL) to supplement the synthetic datset.
        PROS:

            * Accessibility, 
            * True skills-first paradigm
            * Tailored to our needs (format, labels, etc.)
            * Can add a column for quantifying the skills
            * Let's us make models with trainable parameters
        CONS:

            * Not real
            * We will start modelling (or deciding) based on patterns already captured in the LLM - why not just use the LLM
        Criteria:

            * COLUMNS: Job_id, Job_title, Job_description_skills, Job_description_skills_quantied, Resume_id, Resume_name, Resume_skills, Resume_skills_quantified 
        
        
        * Let's refine the prompts and make them perfect - look into litterature on synthetic data


        EXAMPLE PROMPT: 

```python
generated_resumes_list = []

            for job_description in [job_description_list[3]]:
            user_message = HumanMessage(content="""
                You are an AI assistant that helps create resumes for a given job description.
                Generate 2 resumes for each job description so that one resume is an almost perfect match, 
                while the other resume is only slightly relevant. 
                Use a combination of skills, different industry/project work experience, education, 
                and certifications to produce resume data.
                You may add some KPIs to make work experience realistic.
                Do not include any note or explanation of how you generate the resumes. 
            """)
            
            system_message = SystemMessage(content=f"""
                Here is the Job Description (Note that all required skills may not be present in resume 
                and some nonrelevant details can be present). 
                The length of a resume should only be between 200 and 500 words. 
                {job_description}
            """)

            response = llm.invoke([user_message, system_message])
            generated_resumes_list.append(response)
```
            