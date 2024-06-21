from fate.arch import Context
from typing import List, Dict
import logging


logger = logging.getLogger(__name__)


class EncoderDecoder(object):

    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx

    def encode(self,  docs: List[Dict[str, str]], format_template: str):
        pass

    def decode(self,  docs: List[Dict[str, str]], format_template: str ):
        pass

    def inference(self, docs: List[Dict[str, str]], inference_kwargs: dict = {}, format_template: str = None):
        pass