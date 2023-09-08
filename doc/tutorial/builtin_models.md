## Builtin Models
FATE-LLM provide some builtin models, users can use them simply to efficiently train their language models.
To use these models, please read the using tutorial of [ChatGLM-6B Training Guide](./ChatGLM-6B_ds.ipynb) and [GPT2 Training Guide](GPT2-example.ipynb).   
After reading the training tutorial above, it's easy to use other models listing in the following tabular by changing `module_name`, `class_name`, `dataset` to `ModuleName`, `ClassName`, `DatasetName` respectively list below.
  
  

| Model          | ModuleName        | ClassName                         | DataSetName      | 
| -------------- | ----------------- | --------------------------------- | ---------------- |
| Bloom-7B1      | pellm.bloom       | BloomForCausalLM                  | prompt_tokenizer  |                              
| LLaMA-2-7B     | pellm.llama       | LLAMAForCausalLM                  | prompt_tokenizer  |                              
| LLaMA-7B       | pellm.llama       | LLAMAForCausalLM                  | prompt_tokenizer  |                              
| ChatGLM2-6B    | pellm.chatglm     | ChatGLMForConditionalGeneration   | glm_tokenizer    |                              
| ChatGLM-6B     | pellm.chatglm     | ChatGLMForConditionalGeneration   | glm_tokenizer    |                              
| GPT-2          | pellm.gpt2        | GPT2                              | nlp_tokenizer    |                              
| ALBERT         | pellm.albert      | Albert                            | nlp_tokenizer    |                              
| BART           | pellm.bart        | Bart                              | nlp_tokenizer    |                              
| BERT           | pellm.bert        | Bert                              | nlp_tokenizer    |                              
| DeBERTa        | pellm.deberta     | Deberta                           | nlp_tokenizer    |                              
| DistilBERT     | pellm.distilbert  | DistilBert                        | nlp_tokenizer    |                              
| RoBERTa        | pellm.roberta     | Roberta                           | nlp_tokenizer    |                              
