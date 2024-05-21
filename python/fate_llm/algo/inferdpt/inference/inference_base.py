from typing import List


class Inference(object):

    def __init__(self):
        pass

    def inference(self, docs: List[str], inference_kwargs: dict = {}) -> List[str]:
        raise NotImplementedError()