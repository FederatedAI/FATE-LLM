from transformers import AutoTokenizer


def get_seq2seq_tokenizer(model_path):
    from transformers import DataCollatorForSeq2Seq
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    return DataCollatorForSeq2Seq(tokenizer=tokenizer)
