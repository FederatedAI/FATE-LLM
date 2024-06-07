# adopted from https://github.com/huggingface/datasets/blob/main/metrics/rouge/rouge.py


from rouge_score import rouge_scorer


def rouge_l(predictions, references, use_stemmer=False):
    scorer = rouge_scorer.RougeScorer(rouge_types=['rougeL'], use_stemmer=use_stemmer)
    scores = []
    for ref, pred in zip(references, predictions):
        score = scorer.score(ref, pred)
        scores.append(score)

    rouge_l_score = scores[0]['rougeL'].fmeasure
    return rouge_l_score

def doc_to_text(doc):
    if doc["context"]:
        return f"context: {doc['context']}\ninstruction: {doc['instruction']}\nresponse:"
    else:
        return f"instruction: {doc['instruction']}\nresponse:"

"""
def train_load_evalaute_lm():
    pipeline.fit(train_data)
    lm = OTModelLoader().load(path, **args)
    from fate_llm.evaluator import evaluator
    # general case
    evaluator.evaluate(lm, task="dolly_15k", **args)

    # user modified conf
    config = evaluator.get_task_template(task="dolly_15k") # return dict copy of yaml file
    config['dataset_kwargs'] = {"dataset_kwargs":
                                    {"data_files":
                                         {"test": './dolly_15k_test.csv',
                                          "dev": './dolly_15k_dev.csv'}}}
    # may provide arbitrary export path, must be of dir, create temp dir under the given path: {$export_path}/temp_dir
    new_task_dir = evaluator.export_config(config, task="dolly_15k", export_path=None)
    result = evaluator.evalute(lm, task="dolly_15k", include_path=new_task_dir, **args)
    print(result) # dict
    evaluator.delete_config(new_task_dir)
"""