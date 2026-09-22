import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.llm_pick import pick_llm
from models.schema import  JudgeSchema


llm= pick_llm('medium')  # You can change the level to 'low', 'medium', or 'high' as needed
llm_judge = llm.with_structured_output(JudgeSchema)

sql_query = "SELECT * FROM users WHERE age > 30;"  # Example SQL query

prompt = f"""
You are a SQL query judge. Your task is to evaluate the correctness of the following SQL query {sql_query}
This query should only be used for data retrieval and should not modify any data in the database. Please provide your judgment on whether the query is correct or not, and provide any comments if necessary.
This qquery should nopt modify the data using INSERT, UPDATE, DELETE, or any other data modification commands. If the query is correct, respond with 'Yes' and provide any comments if necessary. If the query is incorrect, respond with 'No' and provide comments explaining why it is incorrect.
"""

print(llm_judge.invoke(prompt))