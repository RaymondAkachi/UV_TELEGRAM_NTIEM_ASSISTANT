from .main_graph_state import GraphState
from .RAG.graph import rag_app
from .p_and_c.p_and_c_graph import help_workflow
# from .components.image_creation import TogetherImageGenerator
from .components.watch_sermon import GetSermon
from .components.query_rewriters import query_rewrite
from .components.question_router import question_router
from .appointment_logic.app_graph import app_workflow
### MAIN GRAPH NODES ###

# NODES


async def query_rewriter_node(state: GraphState):
    user_request = state['user_request']
    name = state['name']
    username = state['username']
    chat_id = state['chat_id']
    if isinstance(user_request, list):
        user_request = str(user_request).strip('[').strip(']')
    response = await query_rewrite(user_request, name, username, chat_id)
    is_answerable = response['answerable']
    request_or_answer = response['Response']
    app_state = response['state']
    if is_answerable:
        return {'response': request_or_answer, "output_format": "text", "answered": is_answerable, "app_state": app_state}
    else:
        return {'user_request': request_or_answer, "output_format": "text", "answered": is_answerable, "app_state": app_state}


async def rag_node(state: GraphState):
    user_request = state['user_request']
    rag_validator = state['rag_validator'][0]
    if isinstance(user_request, list):
        user_request = str(user_request).strip('[').strip(']')
    print("ROUTING TO RAG")
    response = await rag_app.ainvoke(
        {"question": str(user_request), 'validators': {
            'validator': rag_validator}})
    result = response['generation']
    return {'response': result, "output_format": "text"}


async def p_and_c_node(state: GraphState):
    user_request = state['user_request']
    validator = state['p_and_c_validators']['validator']
    counselling_validator = state['p_and_c_validators']['counselling_validator']
    app_state = state["app_state"]
    chat_id = state['chat_id']
    if isinstance(user_request, list):
        user_request = str(user_request).strip('[').strip(']')
    print("ROUTUNG TO PRAYER AND COUNSELIING")
    answer = await help_workflow.ainvoke(
        {"request": user_request, 'validators': {
            'prayer_validator': validator, "counselling_validator": counselling_validator},
         "app_state": app_state, "chat_id": chat_id})
    result = answer['response']
    app_state = answer['app_state']
    return {'response': result, 'output_format': "text", "app_state": app_state}


# async def image_creation_node(state: GraphState):
#     user_request = state['user_request']
#     if isinstance(user_request, list):
#         user_request = str(user_request).strip('[').strip(']')
#     print("ROUTING TO IMAGE CREATION")
#     url = TogetherImageGenerator().generate_image(user_request)
#     return {'response': url, "output_format": "image"}


async def watch_sermon_node(state: GraphState):
    user_request = state['user_request']
    if isinstance(user_request, list):
        user_request = str(user_request).strip('[').strip(']')
    print("ROUTING TO WATCH SERMON")
    response = await GetSermon(user_request).get_sermons()
    return {'response': response, 'output_format': "video"}


async def app_node(state: GraphState):
    user_request = state['user_request']
    name = state['name']
    chat_id = state['chat_id']
    username = state['username']
    scheduler = state['scheduler']
    if isinstance(user_request, list):
        user_request = str(user_request).strip('[').strip(']')
    print("ROUTING TO APPOINTMENT NODE")
    app_response = await app_workflow.ainvoke({'user_request': user_request, 'name': name, "username": username,
                                               'chat_id': chat_id, "scheduler": scheduler})
    response = app_response['response']
    return {'response': response, 'output_format': 'text'}


#  EDGES
async def q_router(state: GraphState):
    user_request = state['user_request']
    answered = state['answered']
    if answered:
        return "Answered"
    result = await question_router(user_request)
    return result
