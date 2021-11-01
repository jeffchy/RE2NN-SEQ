from src_seq.ptm.bert_utils import static_bert_embed_decontext, static_bert_embed_aggregate
from src_seq.data import load_slot_dataset

if __name__ == '__main__':
    dataset_name = 'ATIS-BIO'
    # dset = load_slot_dataset(dataset_name, datadir='../../data/')
    # i2t = dset['i2t']
    # static_bert_embed_decontext(i2t, '../../data/{}/bert_decontext.emb'.format(dataset_name))
    static_bert_embed_aggregate(dataset_name, '../../data/{}/bert_aggregate.emb'.format(dataset_name))