import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.llm_pick import pick_llm
from models.schema import AgentSchema, JudgeSchema
from langchain_core.messages import HumanMessage, AIMessage
from utils.database import DatabaseUtil
from langgraph.graph import StateGraph, START, END

#---------------------------AI Agent Node ----------------------------------
def curate_ques(state: AgentSchema) -> AgentSchema:
    llm_obj = pick_llm('low')
    user_question = state.user_question
    curated_question = llm_obj.invoke(f"Please rephrase the following question in a more clear and concise manner: {user_question}").content

    state.curated_ques = curated_question

    state.messages += [HumanMessage(content=f"Curated Question: {curated_question}")]

    return state


def prompt_query_context(state: AgentSchema) -> AgentSchema:
    curated_question = state.curated_ques

    conn_details = {
        "host": os.getenv('host'),
        "port": os.getenv('port'),
        "user": os.getenv('user'),
        "password": os.getenv('password'),
        "database": os.getenv('database')
    }

    obj = DatabaseUtil(conn_details)
    schema_info = obj.schema_details("public")
    
    # Constructing the prompt query for the agent to generate the SQL query
    prompt = f"""
    You are an SQL analyst agent. Your task is to convert the user's natural language 
    query into Postgres SQL query that can be executed on the database. You are provided 
    with the user's original query and the schema details of the database, including
    table names, column names, data types, and sample data for each table so that 
    you can understand the structure of the database and generate an accurate SQL query.
    Unless user explicitly asks for specific number of rows, always limit the output to 10 rows.
    Note - Just generate the SQL query without any explanation or additional text because
    this query will be executed directly on the database. So, the output should be SQL
    ready to be executed without any modifications.  
    
    User's Original Query: {curated_question}

    Database Schema Details:
    {schema_info}
    
    """    

    state.prompt_query_context = prompt

    return state

#---------------------------Generate SQL Node ----------------------------------
def generate_sql(state: AgentSchema) -> AgentSchema:
    prompt = state.prompt_query_context
    llm_obj = pick_llm('medium')
    generated_sql_query = llm_obj.invoke(prompt).content

    state.generated_sql_query = generated_sql_query

    return state

#---------------------------Is Safe  Node ----------------------------------
def is_safe_sql(state: AgentSchema) -> AgentSchema:
    sql_query = state.generated_sql_query
    llm_judge = pick_llm('medium').with_structured_output(JudgeSchema)
    prompt = f"""
    You are a SQL query judge. Your task is to evaluate the correctness of the following SQL query {sql_query}
    This query should only be used for data retrieval and should not modify any data in the database. Please provide your judgment on whether the query is correct or not, and provide any comments if necessary.
    This qquery should nopt modify the data using INSERT, UPDATE, DELETE, or any other data modification commands. If the query is correct, respond with 'Yes' and provide any comments if necessary. If the query is incorrect, respond with 'No' and provide comments explaining why it is incorrect.
    """

    response = llm_judge.invoke(prompt).model_dump()
    state.is_safe= response['answer']
    state.comments = response['comments']

    return state

#---------------------------Cancelled SQL Node ----------------------------------
def cancelled_sql(state: AgentSchema) -> AgentSchema:
    comments = state.comments

    state.final_answer = f"The generated SQL query was deemed unsafe for execution. Comments from the judge: {comments}"
    state.messages += [AIMessage(content=f"Final Answer: {state.final_answer}")]
    return state


#---------------------------Execute SQL Node ----------------------------------
def execute_sql(state: AgentSchema) -> AgentSchema:
    sql_query = state.generated_sql_query
    conn_details = {
        "host": os.getenv('host'),
        "port": os.getenv('port'),
        "user": os.getenv('user'),
        "password": os.getenv('password'),
        "database": os.getenv('database')
    }

    obj = DatabaseUtil(conn_details)
    sql_query_result = obj.execute_query(sql_query)

    state.sql_query_result = sql_query_result

    return state


#---------------------------Final Answer Node ----------------------------------
def represent_final_answer(state: AgentSchema) -> AgentSchema:
    sql_query_result = state.sql_query_result
    llm_obj = pick_llm('low')
    prompt = f"Please provide a concise and clear answer to the user's original question based on the following SQL query result: {sql_query_result}"
    final_answer = llm_obj.invoke(prompt).content

    state.final_answer = final_answer
    state.messages += [AIMessage(content=f"Final Answer: {final_answer}")]

    return state

#---------------------------Final Graph----------------------------------
sql_agent_graph = StateGraph(AgentSchema)
sql_agent_graph.add_node(curate_ques, name="curate_ques")
sql_agent_graph.add_node(prompt_query_context, name="prompt_query_context")
sql_agent_graph.add_node(generate_sql, name="generate_sql")
sql_agent_graph.add_node(is_safe_sql, name="is_safe_sql")
sql_agent_graph.add_node(cancelled_sql, name="cancelled_sql")
sql_agent_graph.add_node(execute_sql, name="execute_sql")
sql_agent_graph.add_node(represent_final_answer, name="represent_final_answer")

# Edges - to connect the above states in the graph
sql_agent_graph.add_edge(START, "curate_ques")
sql_agent_graph.add_edge("curate_ques", "prompt_query_context")
sql_agent_graph.add_edge("prompt_query_context", "generate_sql")
sql_agent_graph.add_edge("generate_sql", "is_safe_sql")

# conditional edges based on the safety of the SQL query
def is_safe_condition(state: AgentSchema) -> str:
    if state.is_safe.lower() == "Yes":
        return "execute_sql"
    else:
        return "cancelled_sql"

sql_agent_graph.add_conditional_edges("is_safe_sql", is_safe_condition, {
    "execute_sql": "execute_sql",
    "cancelled_sql": "cancelled_sql"
})

# sql_agent_graph.add_edge("is_safe_sql", "cancelled_sql")
# sql_agent_graph.add_edge("is_safe_sql", "execute_sql")

sql_agent_graph.add_edge("cancelled_sql", END)
sql_agent_graph.add_edge("execute_sql", "represent_final_answer")
sql_agent_graph.add_edge("represent_final_answer", END)

# Compile the graph to ensure all states and edges are valid
if __name__ == "__main__":
    sql_analyst_agent = sql_agent_graph.compile()

    from IPython.display import display, Image
    img = Image(sql_analyst_agent.get_graph().draw_mermaid_png())
    with open("sql_analyst_agent_graph.png", "wb") as f:
        f.write(img.data)

    input_schema = AgentSchema(
        messages=[],
        user_question="What are the top 5 products by sales in the last quarter?",
        curated_ques="",
        prompt_query_context="",
        generated_sql_query="",
        is_safe="No",
        comments="",
        sql_query_result="",
        final_answer=""
    )

    #execute the agent with the input schema
    sql_analyst_response = sql_analyst_agent.invoke(input_schema)

    print("Final Answer:", sql_analyst_response['messages'][-1].content)
    print('************************************************')
    print(sql_analyst_response['generated_sql_query'])
    print('************************************************')
    print(sql_analyst_response['sql_query_result'])
    print('************************************************')
    print(sql_analyst_response['prompt_query_context'])
