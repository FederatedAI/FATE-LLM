from fate.arch import Context
from fate_llm.algo.inferdpt.inferdpt import InferDPTClient, InferDPTServer
from typing import Union


class InferDPTInit(object):

    def __init__(self, ctx: Context):
        self.ctx = ctx

    def get_inferdpt_inst(self) -> Union[InferDPTClient, InferDPTServer]:
        pass

