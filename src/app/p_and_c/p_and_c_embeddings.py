from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Literal


load_dotenv()


def p_and_c_router():
    # Data model
    class RouteQuery(BaseModel):
        """Route a user query to the most relevant datasource."""

        datasource: Literal["PRAYER", "COUNSELLING"] = Field(
            ...,
            description="Given a user question choose to route it PRAYER, COUNSELLING",
        )

    # LLM with function call
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    structured_llm_router = llm.with_structured_output(RouteQuery)

    # Prompt
    system = """You are an expert at routing user questions.
    Classify the user request:
    - Output "PRAYER" for prayer only.
    - Output "COUNSELLING" for counselling only.

    Examples:
    - "Please pray for me." → PRAYER
    - "I need counselling for stress." → COUNSELLING
    """

    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )

    question_router = route_prompt | structured_llm_router
    return question_router


# async def prayer_and_counselling(request: str, validator, counselling_validator, phone_number, state) -> str:
#     """
#     Create a chain that splits a user's request into separate questions for prayer and/or counselling.

#     Returns:
#         A chain that processes the input and returns a response.
#     """
#     prompt = ChatPromptTemplate.from_template(
#         """"Please split the following request into separate statements for prayer and/or counselling, each on a new line. Each statement must be a complete sentence starting with the verb used in the input (e.g., 'I need,' 'I want'), followed by the service ('prayer' or 'counselling'), and the specific topic if provided. If no topic is specified for a service, use a general statement like 'I need prayer' or 'I need counselling' matching the input verb. If the request doesn’t specify a service but mentions a topic (e.g., 'I need help with anxiety'), assume the topic applies to both prayer and counselling unless indicated otherwise.

# For example:

# **Input**: I need counselling and prayer
# **Output**:
# I need counselling
# I need prayer

# **Input**: I need prayer for my husband
# **Output**:
# I need prayer for my husband

# **Input**: I need counselling for marriage and prayer for anxiety
# **Output**:
# I need counselling for marriage
# I need prayer for anxiety

# **Input**: I want prayer for my health
# **Output**:
# I want prayer for my health

# **Input**: I need help with anxiety
# **Output**:
# I need counselling for anxiety
# I need prayer for anxiety

# Now, split the following request:
# {input}
#         """
#     )
#     llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
#     chain = prompt | llm

#     """
#     Split the input text into separate questions using the provided chain.

#     Args:
#         chain (LLMChain): The chain to process the input.
#         input_text (str): The user's request to split.

#     Returns:
#         list: A list of split questions.
#     """
#     response = await chain.ainvoke(input=request)
#     questions = [q.strip()
#                  for q in str(response.content).split('\n') if q.strip()]
#     total_response = ""

#     for question in questions:
#         if any(word in question.lower() for word in ['pray']):
#             words = question.split()
#             filtered_words = [
#                 word for word in words if "pray" not in word.lower()]
#             filtered_string = " ".join(filtered_words)
#             response, state = await validator.return_help(filtered_string, phone_number, state) + "\n\n"
#             total_response = total_response + response

#         else:
#             words = question.split()
#             filtered_words = [
#                 word for word in words if "counsel" not in word.lower()]
#             filtered_string = " ".join(filtered_words)
#             total_response = total_response + \
#                 await counselling_validator.return_help(filtered_string, phone_number) + "\n\n"

#     return total_response, state


# print(
#     p_and_c_router().invoke(
#         {"question": "I need prayer my cousin is sick and counselling for my marriage"}
#     )
# )
# print(p_and_c_router().invoke({"question": "I need counselling for anxiety"}))
# print(p_and_c_router().invoke(
#     {"question": "I need prayer and counselling for marriage"}).datasource)
