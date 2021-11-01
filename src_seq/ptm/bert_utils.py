from transformers import BertConfig, BertTokenizer, BertModel
from tqdm import tqdm
import torch
import numpy as np
import pickle

from src_seq.utils import pad_dataset_1


def get_max_len(arr):
    """
    :param arr: List[List] list of list varying in length
    :return: the max len of the list
    """
    return max([len(i) for i in arr])


def pad_arr(arr, max_len, pad_element):
    return [i + (max_len - len(i))*[pad_element] for i in arr]


def get_attend_mask(arr, max_len):
    return [[1]*len(i) + [0]*(max_len-len(i)) for i in arr]


def bert_preprocess(dataset, i2t):
    """
    :param dataset: dataset List[List(ids)] with maxlen
    :param i2t: Dict[idx: token]
    :return: bert_padded_input_ids, bert_valid_mask, bert_attend_mask
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=True,
                                              cache_dir='/p300/huggingface/transformers/')

    print("TOKENIZING......")
    pad_id = tokenizer.convert_tokens_to_ids('[PAD]')
    id_querys = []
    bert_valid_masks = []
    for query in tqdm(dataset):
        tokenized_query = ['[CLS]']
        valid_mask = [0] # we do not use the hidden state of CLS and SEP so set it to invalid
        tokens = []
        for token_id in query:
            token = i2t[token_id]
            tokens.append(token)
            if token == '<pad>':
                break

            tokenized_tokens = tokenizer.tokenize(token)
            if len(tokenized_tokens) == 0:
                tokenized_tokens = ['[UNK]']
            tokenized_query += tokenized_tokens
            valid_mask += ([1] + (len(tokenized_tokens)-1)*[0])

        tokenized_query += ['[SEP]']
        valid_mask += [0]
        tokenized_id_query = tokenizer.convert_tokens_to_ids(tokenized_query)

        assert len(tokenized_id_query) == len(valid_mask)
        id_querys.append(tokenized_id_query)
        bert_valid_masks.append(valid_mask)

    # sum of bert valid masks should be equal to length
    max_len = get_max_len(id_querys)
    bert_attend_mask = get_attend_mask(id_querys, max_len)
    bert_padded_input_ids = pad_arr(id_querys, max_len, pad_id)
    bert_valid_mask = pad_arr(bert_valid_masks, max_len, 0)

    return bert_padded_input_ids, bert_attend_mask, bert_valid_mask


def unflatten_with_lengths(obj, L, max_L):
    """
    :param obj: Sum(L) x K (Tensor)
    :param L: B
    :param max_L:
    :return: B x max_L x K
    """

    assert len(obj.size()) == 2
    assert torch.sum(L) == obj.size()[0]
    sum_L, K = obj.size()
    B = len(L)

    temp = torch.zeros(B, max_L, K, device=obj.device).float()
    left = 0
    for i in range(B):
        temp[i,:L[i],:] = obj[left:left+L[i]]
        left = left + L[i]
    return temp


def static_bert_embed_decontext(i2t, save_path):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=True,
                                              cache_dir='/p300/huggingface/transformers/')

    bert = BertModel.from_pretrained('bert-base-uncased', local_files_only=True,
                                          cache_dir='/p300/huggingface/transformers/')

    V = len(i2t)
    embed_mat = []
    all_tokens_id = []
    for idx, token in tqdm(i2t.items()):
        tokenized_tokens = tokenizer.tokenize(token)
        if len(tokenized_tokens) == 0:
            tokenized_tokens = ['[UNK]']
        tokenized_tokens = ['[CLS]'] + tokenized_tokens + ['[SEP]']
        tokenized_id_query = tokenizer.convert_tokens_to_ids(tokenized_tokens)
        all_tokens_id.append(tokenized_id_query)

    max_len = get_max_len(all_tokens_id)
    padded_all_tokens_id = pad_arr(all_tokens_id, max_len, 0)
    attend_mask = get_attend_mask(all_tokens_id, max_len)
    bz = 500
    num_data = len(padded_all_tokens_id)
    right = bz
    while right-bz < num_data:
        batch_ids = padded_all_tokens_id[right-bz: min(right, num_data)]
        batch_masks = attend_mask[right-bz: min(right, num_data)]
        right += bz
        outputs_hiddens = bert(
            input_ids=torch.Tensor(batch_ids).long(),
            attention_mask=torch.Tensor(batch_masks).bool()
        )[0] # bz x L x D_bert
        vecs = outputs_hiddens[:,1,:]  # the first sub token
        embed_mat.append(vecs.detach().numpy())
    embed_mat = np.vstack(embed_mat)
    pickle.dump(embed_mat, open(save_path, 'wb'))


def static_bert_embed_aggregate(dataset_name, save_path):
    from src_seq.data import load_slot_dataset
    dset = load_slot_dataset(dataset_name, datadir='../../data/')
    t2i, i2t, s2i, i2s = dset['t2i'], dset['i2t'], dset['s2i'], dset['i2s']
    query_train, slot_train = dset['query_train'], dset['intent_train']
    query_dev, slot_dev = dset['query_dev'], dset['intent_dev']
    query_test, slot_test = dset['query_test'], dset['intent_test']
    i2t[len(i2t)] = '<pad>'
    t2i['<pad>'] = len(i2t) - 1
    seq_max_len = 30
    train_query, _, train_lengths = pad_dataset_1(query_train, seq_max_len, t2i['<pad>'])
    dev_query, _, dev_lengths = pad_dataset_1(query_dev, seq_max_len, t2i['<pad>'])
    test_query, _, test_lengths = pad_dataset_1(query_test, seq_max_len, t2i['<pad>'])
    bert = BertModel.from_pretrained('bert-base-uncased', local_files_only=True,
                                     cache_dir='/p300/huggingface/transformers/')

    datasets_query = [train_query, dev_query, test_query]
    datasets_length = [train_lengths, dev_lengths, test_lengths]
    embed_list = [[] for i in range(len(i2t)-1)]

    for _query, _length in zip(datasets_query, datasets_length):
        bert_padded_input_ids, bert_attend_mask, bert_valid_mask = bert_preprocess(_query, i2t)

        bz = 1000
        num_data = len(bert_padded_input_ids)
        right = bz
        while right-bz < num_data:
            batch_ids = bert_padded_input_ids[right-bz: min(right, num_data)]
            batch_atten_masks = bert_attend_mask[right-bz: min(right, num_data)]
            batch_valid_masks = bert_valid_mask[right-bz: min(right, num_data)]
            batch_lengths = _length[right-bz: min(right, num_data)]
            batch_query = _query[right-bz: min(right, num_data)]
            right += bz

            outputs_hiddens = bert(
                input_ids=torch.Tensor(batch_ids).long(),
                attention_mask=torch.Tensor(batch_atten_masks).bool()
            )[0] # bz x L x D_bert
            valid_hiddens = outputs_hiddens[torch.Tensor(batch_valid_masks).bool()] # Sum(L) x 768
            L = max(batch_lengths)
            padded_bert_hidden = unflatten_with_lengths(
                valid_hiddens,
                torch.Tensor(batch_lengths).long(),
                max(batch_lengths))  # B x max_L x 768

            B, _, D_bert = padded_bert_hidden.size()

            for i in tqdm(range(B)):
                query = batch_query[i]

                for j in range(L):
                    token_id = query[j]
                    token = i2t[token_id]
                    if token == '<pad>':
                        break

                    embed_list[token_id].append(padded_bert_hidden[i][j].cpu().detach().numpy())

    for i in range(len(embed_list)):
        if len(embed_list[i]) == 0:
            embed_list[i] = np.zeros(768,)
        else:
            embed_list[i] = np.mean(embed_list[i], 0)

    embed_mat = np.vstack(embed_list)
    pickle.dump(embed_mat, open(save_path, 'wb'))


def load_bert_embed(args):
    dataset = args.dataset
    embed_type = args.bert_init_embed
    embed = pickle.load(open('../data/{}/bert_{}.emb'.format(dataset, embed_type), 'rb'))
    return embed
