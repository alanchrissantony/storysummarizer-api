from pydantic import BaseModel

class SummarizationRequest(BaseModel):
    text: str
    method: str  
    num_sentences: int
