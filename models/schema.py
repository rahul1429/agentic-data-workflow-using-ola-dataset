from pydantic import BaseModel, Field
from typing import Annotated, Literal
from operator import add

class AgentSchema(BaseModel):
    messages : Annotated[list, add] = Field(..., description="List of messages")
    user_question : str = Field(..., description="User's question")
    curated_ques : str = Field(..., description="Currated user question")
    prompt_query_context : str = Field(..., description="A detailed Prompt with SQL DB context")
    is_safe : Literal["Yes", "No"] = Field(..., description="Indicates if the query is safe")
    comments : str = Field(..., description="Comments from the judge on the query")
    generated_sql_query : str = Field(..., description="Generated SQL query")
    sql_query_result : str = Field(..., description="Result of the generated SQL query")
    final_answer : str = Field(..., description="Final answer to the user question")
    

class JudgeSchema(BaseModel):
    answer: Literal["Yes", "No"] = Field(..., description="Indicates if the answer is correct")
    comments: str = Field(..., description="Additional comments from the judge on the answer")

class ETLAgentSchema(BaseModel):
    messages : Annotated[list, add] = Field(..., description="List of messages to be processed by the ETL agent")