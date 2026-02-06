## Builtin PELLM Models
FATE-LLM provide some builtin pellm models, users can use them simply to efficiently train their language models.
To use these models, please read the using tutorial of [ChatGLM-6B Training Guide](./ChatGLM3-6B_ds.ipynb).   
After reading the training tutorial above, it's easy to use other models listing in the following tabular by changing `module_name`, `class_name`, `dataset` list below.
  
  

| Model          | ModuleName        | ClassName     | DataSetName     | 
| -------------- | ----------------- | --------------| --------------- |                 
| Qwen2          | pellm.qwen        | Qwen          | prompt_dataset  |                              
| Bloom-7B1      | pellm.bloom       | Bloom         | prompt_dataset  |                              
| OPT-6.7B       | pellm.opt         | OPT           | prompt_dataset  |                              
| LLaMA-2-7B     | pellm.llama       | LLaMa         | prompt_dataset  |                              
| LLaMA-7B       | pellm.llama       | LLaMa         | prompt_dataset  |                              
| ChatGLM3-6B    | pellm.chatglm     | ChatGLM       | prompt_dataset  |                              
| GPT-2          | pellm.gpt2        | GPT2CLM       | prompt_dataset  |                              
| GPT-2          | pellm.gpt2        | GPT2          | seq_cls_dataset |                              
| ALBERT         | pellm.albert      | Albert        | seq_cls_dataset |                              
| BART           | pellm.bart        | Bart          | seq_cls_dataset |                              
| BERT           | pellm.bert        | Bert          | seq_cls_dataset |                              
| DeBERTa        | pellm.deberta     | Deberta       | seq_cls_dataset |                              
| DistilBERT     | pellm.distilbert  | DistilBert    | seq_cls_dataset |                              
| RoBERTa        | pellm.roberta     | Roberta       | seq_cls_dataset |                              
