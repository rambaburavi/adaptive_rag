from .retrieve import retrieve
from .rerank import rerank
from .web_search import web_search
from .answer_directly import answer_directly
from .decompose import decompose

ACTIONS = {
    "retrieve": retrieve,
    "rerank": rerank,
    "web_search": web_search,
    "answer_directly": answer_directly,
    "decompose": decompose,
}