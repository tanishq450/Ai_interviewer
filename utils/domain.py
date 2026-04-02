class DetectDomain:
    def __init__(self):
        self.logger = loguru.logger
        self.model = ModelLoader().load_llm()

    def detect_domain(self,resume_text:str):
        try:
            self.logger.info("Detecting domain")
            prompt = f"""
            You are an expert in analyzing resumes and identifying the domain of the candidate.
            Analyze the following resume text and identify the domain of the candidate.
            Return the domain in the following format:
            {{
                "domain": "<domain>",
                "topic": "<topic>",
                "difficulty": "<difficulty>"
            }}
            Resume Text:
            {resume_text}
            """
            response = self.model.invoke(prompt)
            self.logger.info("Domain detected successfully")
            return response
        except Exception as e:
            self.logger.error(f"Error detecting domain: {e}")
            return None 


    def detect_difficulty(self,resume_text:str):
        try:
            self.logger.info("Detecting difficulty")
            prompt = f"""
            You are an expert in analyzing resumes and identifying the difficulty level of the candidate.
            Analyze the following resume text and identify the difficulty level of the candidate.
            Return the difficulty level in the following format:
            {{
                "difficulty": "<difficulty>"
            }}
            Resume Text:
            {resume_text}
            """
            response = self.model.invoke(prompt)
            self.logger.info("Difficulty detected successfully")
            return response
        except Exception as e:
            self.logger.error(f"Error detecting difficulty: {e}")
            return None 
        



