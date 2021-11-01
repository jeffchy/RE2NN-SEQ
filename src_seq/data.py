from torch.utils.data import Dataset
import os
import numpy as np
import pickle
import sys
sys.path.append('../../')
sys.path.append('../')
from src_seq.load_data_and_rules import *
from src_seq.utils import mkdir
from src_seq.utils import divide_list_into_N, set_seed
from src_seq.tools.timer import Timer
from copy import copy
from src_seq.rule_utils.rule_pre_parser import rule_pre_parser, load_slot_rule_from_lines
from src_seq.utils import even_select_from_total_number, even_select_from_portion
from src_seq.analysis.utils import split_dev, split_dev_marryup
from collections import Counter
import fasttext.util
from tqdm import tqdm
from src_seq.ptm.bert_utils import bert_preprocess
timer = Timer()


def create_vocabs(iterable, mode):
    assert mode in ['labels', 'texts']
    vocab = Counter()
    if mode == 'labels':
        vocab = vocab + Counter(list(iterable))
    else:
        for instance in iterable:
            vocab += Counter(instance)

    vocab_list = list(vocab.keys())
    i2v = {idx: vocab for idx, vocab in enumerate(vocab_list)}
    v2i = {vocab: idx for idx, vocab in enumerate(vocab_list)}

    return i2v, v2i


def make_glove_embed(dataset_path, i2t, glove_path, embed_dim=100):
    glove = {}
    vecs = [] # use to produce unk

    i = 0
    # load glove
    with open(os.path.join(glove_path, 'glove.6B.{}d.txt'.format(embed_dim)),
              'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            i += 1
            # if i > 100000:
            #     break
            split_line = line.split()
            word = split_line[0]
            embed_str = split_line[1:]
            try:
                # ignore error
                embed_float = [float(i) for i in embed_str]
            except:
                continue

            if word not in glove:
                glove[word] = embed_float
                vecs.append(embed_float)

    unk = np.zeros(embed_dim)
    print("error lines: {}".format(i-len(vecs)))

    print("loading glove to task vocab")
    # load glove to task vocab
    embed = []
    for i in tqdm(i2t):
        word = i2t[i].lower()
        if word in glove:
            embed.append(glove[word])
        else:
            embed.append(unk)

    final_embed = np.array(embed, dtype=np.float)
    num_error = np.sum(final_embed[final_embed == 0])
    print("NUM ERROR: {}".format(num_error))
    pickle.dump(final_embed, open(os.path.join(dataset_path, 'glove.{}.emb'.format(embed_dim)), 'wb'))
    print("SAVED")
    return


def load_glove_embed(dataset_path, embed_dim):
    """
    :param config:
    :return the numpy array of embedding of task vocabulary: V x D:
    """
    return pickle.load(open(os.path.join(dataset_path, 'glove.{}.emb'.format(embed_dim)), 'rb'))


def load_fasttext_embed(dataset_path, embed_dim):

    return pickle.load(open(os.path.join(dataset_path, 'fasttext.{}.emb'.format(embed_dim)), 'rb'))


def make_fasttext_embed(dataset_path, i2t, fasttext_path='', embed_dim=100):
    assert embed_dim in [100, 300]
    if 'ZH' not in dataset_path:
        lang = 'en'
    else:
        lang = 'zh'

    print("Loading FastText")
    ft = fasttext.load_model(os.path.join(fasttext_path, 'cc.{}.{}.bin'.format(lang, embed_dim)))
    embed = []
    for i in tqdm(i2t):
        word = i2t[i].lower()
        embed.append(ft.get_word_vector(word))
    final_embed = np.array(embed, dtype=np.float)

    pickle.dump(final_embed, open(os.path.join(dataset_path, 'fasttext.{}.emb'.format(embed_dim)), 'wb'))
    print("SAVED")

    return


class SlotBatchDatasetNoRE(Dataset):

    def __init__(self, query, lengths, slot, args, s2i, portion=1, dset='train'):
        assert dset in ['train', 'dev', 'test']
        assert len(query) == len(slot)


        if portion == 1.0 or portion == 0.0:
            self.dataset = query
            self.slot = slot
            self.lengths = lengths

        else:
            if portion > 1: # in this case, the portion is the shots
                size = int(portion)
            else:
                size = int(portion*len(query))

            if dset in ['dev']:
                size = max(size, 200)  # dev set at least 200 sample
            idxs = even_select_from_total_number(len(query), size, seed=args.seed)

            self.dataset = list(np.array(query)[idxs])
            self.slot = list(np.array(slot)[idxs])
            self.lengths = list(np.array(lengths)[idxs])

    def __getitem__(self, idx):

        return {
            'x': np.array(self.dataset[idx], dtype=np.int),
            's': np.array(self.slot[idx], dtype=np.int),
            'l': np.array(self.lengths[idx], dtype=np.int),
        }

    def __len__(self):
        return len(self.dataset)



class SlotBatchDataset(Dataset):

    def __init__(self, query, lengths, slot, args, s2i, portion=1, dset='train'):
        assert dset in ['train', 'dev', 'test']
        assert len(query) == len(slot)
        from src_seq.RE import predict_by_RE
        train_pred, dev_pred, test_pred, train_score, dev_score, test_score = predict_by_RE(args)
        if dset == 'dev':
            re_out = dev_score
            re_pred = dev_pred
        elif dset == 'test':
            re_out = test_score
            re_pred = test_pred
        else:
            re_out = train_score
            re_pred = train_pred

        if args.use_unlabel:
            re_slot = [re_pred[i].numpy() for i in range(len(re_pred))]
            slot = re_slot if (dset != 'test') else slot

        if portion == 1.0 or portion == 0.0:
            self.dataset = query
            self.slot = slot
            self.lengths = lengths
            try:
                self.re = list(re_out.numpy())
            except:
                self.re = re_out
        else:
            if portion > 1: # in this case, the portion is the shots
                size = int(portion)
            else:
                size = int(portion*len(query))

            if dset in ['dev']:
                size = max(size, 200)  # dev set at least 200 sample
            idxs = even_select_from_total_number(len(query), size, seed=args.seed)

            self.dataset = list(np.array(query)[idxs])
            self.slot = list(np.array(slot)[idxs])
            self.lengths = list(np.array(lengths)[idxs])
            self.re = list(re_out.numpy()[idxs])

    def __getitem__(self, idx):

        return {
            'x': np.array(self.dataset[idx], dtype=np.int),
            's': np.array(self.slot[idx], dtype=np.int),
            'l': np.array(self.lengths[idx], dtype=np.int),
            're': np.array(self.re[idx], dtype=np.float32),
        }

    def __len__(self):
        return len(self.dataset)


class MarryUpSlotBatchDataset(Dataset):

    def __init__(self, query, lengths, slot, args, s2i, portion=1.0, dset='train'):

        assert dset in ['train', 'dev', 'test']
        assert len(query) == len(slot)
        from src_seq.RE import predict_by_RE
        train_pred, dev_pred, test_pred, train_score, dev_score, test_score = predict_by_RE(args)
        if dset == 'dev':
            re_out = dev_score
            re_pred = dev_pred
        elif dset == 'test':
            re_out = test_score
            re_pred = test_pred
        else:
            re_out = train_score
            re_pred = train_pred

        re_slot = [re_pred[i].numpy() for i in range(len(re_pred))]
        slot = re_slot if (args.use_unlabel and dset != 'test') else slot

        if portion == 1.0:
            self.dataset = query
            self.lengths = lengths
            self.slot = slot
            try:
                self.re = list(re_out.numpy())
            except:
                self.re = re_out

        else:
            if portion > 1: # in this case, the portion is the shots
                size = int(portion)
            else:
                size = int(portion*len(query))

            if dset in ['dev']:
                size = max(size, 200)  # dev set at least 200 sample
            idxs = even_select_from_total_number(len(query), size, seed=args.seed)
            self.dataset = list(np.array(query)[idxs])
            self.slot = list(np.array(slot)[idxs])
            self.lengths = list(np.array(lengths)[idxs])
            self.re = list(re_out.numpy()[idxs])

    def __getitem__(self, idx):

        return {
            'x': np.array(self.dataset[idx], dtype=np.int),
            's': np.array(self.slot[idx], dtype=np.int),
            'l': np.array(self.lengths[idx], dtype=np.int),
            're': np.array(self.re[idx], dtype=np.float32),
        }

    def __len__(self):
        return len(self.dataset)


class SlotBatchDatasetPTM(Dataset):

    def __init__(self, query, lengths, slot, args, i2t, portion=1, dset='train'):
        assert dset in ['train', 'dev', 'test']
        assert len(query) == len(slot)
        from src_seq.RE import predict_by_RE
        train_pred, dev_pred, test_pred, train_score, dev_score, test_score = predict_by_RE(args)
        if dset == 'dev':
            re_out = dev_score
            re_pred = dev_pred
        elif dset == 'test':
            re_out = test_score
            re_pred = test_pred
        else:
            re_out = train_score
            re_pred = train_pred

        if args.use_unlabel and dset != 'test':
            re_slot = [re_pred[i].numpy() for i in range(len(re_pred))]
            # idxs = [i for i in range(len(re_slot)) if re_slot[i].sum() > 0]
            # re_slot = [re_slot[i] for i in idxs]
            # query = [query[i] for i in idxs]
            # lengths = [lengths[i] for i in idxs]
            slot = re_slot

        if portion == 1.0 or portion == 0.0:
            self.dataset = query
            self.slot = slot
            self.lengths = lengths
            try:
                self.re = list(re_out.numpy())
            except:
                self.re = re_out
        else:
            if portion > 1: # in this case, the portion is the shots
                size = int(portion)
            else:
                size = int(portion*len(query))

            if dset in ['dev']:
                size = max(size, 200)  # dev set at least 200 sample
            idxs = even_select_from_total_number(len(query), size, seed=args.seed)

            self.dataset = list(np.array(query)[idxs])
            self.slot = list(np.array(slot)[idxs])
            self.lengths = list(np.array(lengths)[idxs])
            self.re = list(re_out.numpy()[idxs])

        # For bert tokenizer
        self.bert_padded_input_ids, self.bert_attend_mask, self.bert_valid_mask = bert_preprocess(self.dataset, i2t)

    def __getitem__(self, idx):

        return {
            'x':  np.array(self.dataset[idx], dtype=np.int),
            'x_bert': np.array(self.bert_padded_input_ids[idx], dtype=np.int),
            'attend_mask': np.array(self.bert_attend_mask[idx], dtype=np.int),
            'valid_mask': np.array(self.bert_valid_mask[idx], dtype=np.int),
            's': np.array(self.slot[idx], dtype=np.int),
            'l': np.array(self.lengths[idx], dtype=np.int),
            're': np.array(self.re[idx], dtype=np.float32),
        }

    def __len__(self):
        return len(self.dataset)


def load_slot_dataset(dataset, datadir='../data/'):
    assert dataset in ['ATIS-BIO', 'ATIS-ZH-BIO', 'SNIPS-BIO']
    return pickle.load(open('{}{}/dataset.pkl'.format(datadir, dataset), 'rb'))


def create_slot_dataset(dataset_name):

    timer.start()
    print('LOADING DATASET')

    if dataset_name == 'ATIS-BIO':
        DATA_DIR = '../data/'
        t2i, i2t, s2i, i2s, dicts = load_dict_ATIS(DATA_DIR='{}/{}'.format(DATA_DIR, dataset_name))
        query_train, slots_train,  intent_train = load_data_ATIS(mode='train', DATA_DIR='{}/ATIS'.format(DATA_DIR))
        query_dev, slots_dev, intent_dev = load_data_ATIS(mode='dev', DATA_DIR='{}/ATIS'.format(DATA_DIR))
        query_test, slots_test, intent_test= load_data_ATIS(mode='test', DATA_DIR='{}/ATIS'.format(DATA_DIR))
        print('CREATING EMBED')
        make_glove_embed('../data/{}'.format(dataset_name), i2t, '../data/emb/glove.6B')
        dataset = {
            't2i': t2i, 'i2t': i2t, 's2i': s2i, 'i2s': i2s,
            'query_train': query_train, 'intent_train': slots_train,
            'query_dev': query_dev, 'intent_dev': slots_dev,
            'query_test': query_test, 'intent_test': slots_test,
        }
        print('SAVING DATASET')
        pickle.dump(dataset, open('../data/{}/dataset.pkl'.format(dataset_name), 'wb'))
        return
    elif dataset_name == 'SNIPS-BIO':
        res = load_SNIPS_dataset()
    else:
        res = load_BIO_dataset(dataset_name)

    data = res['data']
    tags = list(data['tags'])
    texts = list(data['text'])

    timer.stop()

    timer.start()
    print('CREATING VOCABS')
    i2s, s2i = create_vocabs(tags, 'texts')
    i2t, t2i = create_vocabs(texts, 'texts')

    timer.stop()

    timer.start()
    print('CREATING EMBED FILE')
    if 'ZH' in dataset_name:
        make_fasttext_embed('../data/{}'.format(dataset_name), i2t, '../data/emb/fasttext')
    else:
        make_glove_embed('../data/{}'.format(dataset_name), i2t, '../data/emb/glove.6B')

    timer.stop()

    timer.start()
    print('TRANSFORMING TO INDEX')
    data = data.groupby('mode')
    train, dev, test = data.get_group('train'), data.get_group('dev'), data.get_group('test')
    timer.stop()

    def to_query_slot(dataset):
        tags = list(dataset['tags'])
        texts = list(dataset['text'])
        tags = [[s2i[j] for j in i] for i in tags]
        query = [[t2i[j] for j in i] for i in texts]
        return tags, query

    slot_train, query_train = to_query_slot(train)
    slot_dev, query_dev = to_query_slot(dev)
    slot_test, query_test = to_query_slot(test)

    timer.start()
    print('SAVING DATASET')
    dataset = {
        't2i': t2i, 'i2t': i2t, 's2i': s2i, 'i2s': i2s,
        'query_train': query_train, 'intent_train': slot_train,
        'query_dev': query_dev, 'intent_dev': slot_dev,
        'query_test': query_test, 'intent_test': slot_test,
    }
    pickle.dump(dataset, open('../data/{}/dataset.pkl'.format(dataset_name), 'wb'))
    timer.stop()


def read_rules(dataset_name, rule_name='bio.rules.config'):


    if dataset_name == 'ATIS-BIO':
        # rules = load_rule_slot(os.path.join(MITR_BIO_PATH, 'bio.rules.21.config'))
        complete_lines = rule_pre_parser(os.path.join(ATIS_BIO_PATH, 'bio.rules.config'))
        rules = load_slot_rule_from_lines(complete_lines)
    elif dataset_name == 'ATIS-ZH-BIO':
        complete_lines = rule_pre_parser(os.path.join(ATIS_ZH_BIO_PATH, rule_name))
        rules = load_slot_rule_from_lines(complete_lines)
    elif dataset_name == 'SNIPS-BIO':
        complete_lines = rule_pre_parser(os.path.join(SNIPS_PATH, rule_name))
        rules = load_slot_rule_from_lines(complete_lines)
    else:
        raise ValueError('WRONG DATASET NAME')

    return rules


def create_embedding_from_data(dataset, embed_dim, embed_type='glove'):
    data = load_slot_dataset(dataset)
    if embed_type == 'glove':
        make_glove_embed('../data/{}'.format(dataset), data['i2t'],  '../data/emb/glove.6B', embed_dim=embed_dim)
    if embed_type == 'fasttext':
        make_fasttext_embed('../data/{}'.format(dataset), data['i2t'], '../data/emb/fasttext',  embed_dim=embed_dim)


if __name__ == '__main__':
    set_seed(0)

    # create_slot_dataset('SNIPS-BIO')
    # create_slot_dataset('ATIS-BIO')
    # create_slot_dataset('ATIS-ZH-BIO')


