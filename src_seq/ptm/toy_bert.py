from transformers import BertModel, BertConfig, BertTokenizer

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=True, cache_dir='/p300/huggingface/transformers/')
    model = BertModel.from_pretrained('bert-base-uncased', local_files_only=True, cache_dir='/p300/huggingface/transformers/')
    inputs = tokenizer("cutest cat", return_tensors="pt")
    inputs1 = tokenizer.tokenize("cutest")
    input_ids1 = tokenizer.convert_tokens_to_ids(inputs1)
    outputs = model(**inputs)
    params = model.parameters()
    print(outputs[0])


# [1] Titan, RTX
# [2] [CLS] Titan, R, ##T, ##X, [SEP]
# [3] Titan), R

