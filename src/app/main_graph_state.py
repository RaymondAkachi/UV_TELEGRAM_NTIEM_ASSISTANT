from typing_extensions import TypedDict
from typing import Dict, Union, List


class GraphState(TypedDict):
    """state on main graph"""
    output_format:  str
    response: Union[str, Dict, List]
    p_and_c_validators: Dict
    rag_validator: List
    name: str
    username: str
    chat_id: str
    user_request: str
    scheduler: List
    answered: bool
    app_state: str
