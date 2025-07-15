from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from dotenv import load_dotenv
from sqlalchemy import create_engine
import os
from typing import Any
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Annotated, Literal
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from langchain_teddynote.messages import random_uuid, invoke_graph, stream_graph
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError


# load the environment variables, may require .env file
load_dotenv()

# connect to the database   
def connect_db():
    server = os.getenv("DB_SERVER")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    port = os.getenv("DB_PORT")
    database = "MIMIC3_CDM"
    connection_string = f"mssql+pymssql://{user}:{password}@{server}:{port}/{database}"
    engine = create_engine(connection_string)
    return engine

engine = connect_db()
db = SQLDatabase(engine=engine,
                 ignore_tables = ["cohort_definition", "cost", "device_exposure", "location",
        "metadata", "note_nlp", "payer_plan_period", "source_to_concept_map","achilles_results","achilles_results_derived","achilles_results_dist","ACHILLES_ANALYSIS","achilles_heel_results"])


MODEL_NAME="gpt-4o"


# handling tool error
def handle_tool_error(state) -> dict:
    # error information
    error = state.get("error")
    # searching for the tool calls
    tool_calls = state["messages"][-1].tool_calls
    # ToolMessage wrraping and return
    return {
        "messages": [
            ToolMessage(
                content=f"Here is the error: {repr(error)}\n\nPlease fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


# creating tool node with fallback
def create_tool_node_with_fallback(tools: list) -> RunnableWithFallbacks[Any, dict]:
    """
    Create a ToolNode with a fallback to handle errors and surface them to the agent.
    """
    # for error handling
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )
    


# SQLDatabaseToolkit creation
toolkit = SQLDatabaseToolkit(db=db, llm=ChatOpenAI(model=MODEL_NAME))

# SQLDatabaseToolkit available tools
tools = toolkit.get_tools()

# table list tool
list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")

# getting schema
get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")




# Query execution tool
@tool
def db_query_tool(query: str) -> str:
    """
    Run SQL queries against a database and return results
    Returns an error message if the query is incorrect
    If an error is returned, rewrite the query, check, and retry
    """
    # execute the query, return in string
    result = db.run_no_throw(query)

    
    if not result:
        return "Error: Query failed. Please rewrite your query and try again."
    
    return result




# double check the query for sql syntax
query_check_system = """You are a SQL expert with a strong attention to detail.
Double check the MS-SQL query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

You will call the appropriate tool to execute the query after running this check."""

# prompt for query check
query_check_prompt = ChatPromptTemplate.from_messages(
    [("system", query_check_system), ("placeholder", "{messages}")]
)

# Query Checker chain
query_check = query_check_prompt | ChatOpenAI(
    model=MODEL_NAME, temperature=0
).bind_tools([db_query_tool], tool_choice="db_query_tool")




# define state for this agent
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# 새로운 그래프 정의
workflow = StateGraph(State)


# 1st too call function
def first_tool_call(state: State) -> dict[str, list[AIMessage]]:
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "sql_db_list_tables",
                        "args": {},
                        "id": "initial_tool_call_abc123",
                    }
                ],
            )
        ]
    }


# checking the query
def model_check_query(state: State) -> dict[str, list[AIMessage]]:
    """
    Use this tool to check that your query is correct before you run it
    """
    return {"messages": [query_check.invoke({"messages": [state["messages"][-1]]})]}


# 1st calling
workflow.add_node("first_tool_call", first_tool_call)

# list tables
workflow.add_node(
    "list_tables_tool", create_tool_node_with_fallback([list_tables_tool])
)
workflow.add_node("get_schema_tool", create_tool_node_with_fallback([get_schema_tool]))

# model getting schema
model_get_schema = ChatOpenAI(model=MODEL_NAME, temperature=0).bind_tools(
    [get_schema_tool]
)
workflow.add_node(
    "model_get_schema",
    lambda state: {
        "messages": [model_get_schema.invoke(state["messages"])],
    },
)


# final answer formation
class SubmitFinalAnswer(BaseModel):
    """submit final answer"""

    final_answer: str = Field(..., description="The final answer to the user")


# prompt specific for MIMIC3 CDM
QUERY_GEN_INSTRUCTION = """You are a MS-SQL expert with a strong attention to detail.

You can define MS-SQL queries, analyze queries results and interpretate query results to response an answer. the database is MIMIC3 CDM.

Read the messages bellow and identify the user question, table schemas, query statement and query result, or error if they exist.

1. If there's not any query result that make sense to answer the question, create a syntactically correct MS-SQL query to answer the user question. DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

2. If you create a query, response ONLY the query statement. For example, "SELECT id, name FROM pets;"

3. If a query was already executed, but there was an error. Response with the same error message you found. For example: "Error: Pets table doesn't exist"

4. if the query ask for the age, you should calculate the age by the following formula: age = (admission time) in visit_occurrence table minus (year of birth) in person table.

5. When answering provide concept_name instead of concept_id.

6. If definition is not in the concept_name, formulate appropriate definition with your knowledge.

7. If a query was already executed successfully interpretate the response and answer the question following this pattern: Answer: <<question answer>>. For example: "Answer: There three cats registered as adopted"
"""

query_gen_prompt = ChatPromptTemplate.from_messages(
    [("system", QUERY_GEN_INSTRUCTION), ("placeholder", "{messages}")]
)
query_gen = query_gen_prompt | ChatOpenAI(model=MODEL_NAME, temperature=0).bind_tools(
    [SubmitFinalAnswer, model_check_query]
)


# conditional edge
def should_continue(state: State) -> Literal[END, "correct_query", "query_gen"]:
    messages = state["messages"]

    last_message = messages[-1]
    if last_message.content.startswith("Answer:"):
        return END
    if last_message.content.startswith("Error:"):
        return "query_gen"
    else:
        return "correct_query"


# query generation node
def query_gen_node(state: State):
    message = query_gen.invoke(state)

    # if the wrong tool was called, return the error message
    tool_messages = []
    message.pretty_print()
    if message.tool_calls:
        for tc in message.tool_calls:
            if tc["name"] != "SubmitFinalAnswer":
                tool_messages.append(
                    ToolMessage(
                        content=f"Error: The wrong tool was called: {tc['name']}. Please fix your mistakes. Remember to only call SubmitFinalAnswer to submit the final answer. Generated queries should be outputted WITHOUT a tool call.",
                        tool_call_id=tc["id"],
                    )
                )
    else:
        tool_messages = []
    return {"messages": [message] + tool_messages}


# query generation node
workflow.add_node("query_gen", query_gen_node)

# query check node
workflow.add_node("correct_query", model_check_query)

# query execution node
workflow.add_node("execute_query", create_tool_node_with_fallback([db_query_tool]))

# edge between nodes
workflow.add_edge(START, "first_tool_call")
workflow.add_edge("first_tool_call", "list_tables_tool")
workflow.add_edge("list_tables_tool", "model_get_schema")
workflow.add_edge("model_get_schema", "get_schema_tool")
workflow.add_edge("get_schema_tool", "query_gen")
workflow.add_conditional_edges(
    "query_gen",
    should_continue,
)
workflow.add_edge("correct_query", "execute_query")
workflow.add_edge("execute_query", "query_gen")

# complie workflow
app = workflow.compile(checkpointer=MemorySaver())


def run_graph(
    message: str, recursive_limit: int = 30, node_names=[], stream: bool = False
):
    # config setting
    config = RunnableConfig(
        recursion_limit=recursive_limit, configurable={"thread_id": random_uuid()}
    )

    # question input
    inputs = {
        "messages": [HumanMessage(content=message)],
    }

    try:
        if stream:
            
            stream_graph(app, inputs, config, node_names=node_names)
        else:
            invoke_graph(app, inputs, config, node_names=node_names)
        output = app.get_state(config).values
        return output
    except GraphRecursionError as recursion_error:
        print(f"GraphRecursionError: {recursion_error}")
        output = app.get_state(config).values
        return output

def invoke_graph(app, inputs, config, node_names=[]):
    app.invoke(inputs, config)

def stream_graph(app, inputs, config, node_names=[]):
    for chunk in app.stream(inputs, config):
        print(chunk)

def interactive_sql():
    print("\nWelcome to the SQL Assistant! Type 'exit' to quit.")

    while True:
        try:
            query = input("\nWhat would you like to know? ")
            if query.lower() in ['exit', 'quit']:
                print("\nThank you for using SQL Assistant!")
                break

            output = run_graph(query)
            # Extract the final answer from the output
            if 'messages' in output and output['messages']:
                final_message = output['messages'][-1]
                if hasattr(final_message, 'content'):
                    print("\nResponse:", final_message.content)
                else:
                    print("\nResponse: No clear answer found.")
            else:
                print("\nNo response generated.")

        except KeyboardInterrupt:
            print("\nThank you for using SQL Assistant!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            print("Please try again with a different query.")


if __name__ == "__main__":
    interactive_sql()




