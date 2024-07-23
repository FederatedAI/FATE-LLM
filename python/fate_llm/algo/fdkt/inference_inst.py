from fate_llm.inference.api import APICompletionInference


def init(api_url: str, model_name: str, api_key: str = 'EMPTY', api_timeout=3600):
    return APICompletionInference(
        api_url=api_url,
        model_name=model_name,
        api_key=api_key,
        api_timeout=api_timeout
    )