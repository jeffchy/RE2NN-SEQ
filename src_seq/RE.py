import torch
from src_seq.data import load_slot_dataset, SlotBatchDatasetNoRE
# from data import load_slot_dataset, SlotBatchDataset1, SlotBatchDataset
from torch.utils.data import DataLoader
from src_seq.wfa.fsa_to_tensor import dfa_to_tensor_slot_new, dfa_to_tensor_slot_independent, dfa_to_tensor_slot_single_wildcard, dfa_to_tensor_slot_single
from src_seq.farnn.model_onehot import FARNN_S_O, FARNN_S_O_I,FARNN_S_O_I_S
from tqdm import tqdm
from copy import copy
from src_seq.create_logic_mat_bias import create_mat_priority_MITR
import os, pickle
from src_seq.utils import flatten, Logger, len_stats, pad_dataset_1, set_seed, load_pkl
from src_seq.metrics.metrics import get_ner_fmeasure, eval_seq_token


def get_RE_prediction(dataloader, model, args, o_idx=0, i2s=None):
    tensors = []
    tensors_score = []
    all_pred_label = []
    all_true_label = []
    data = tqdm(dataloader)
    model.eval()
    with torch.no_grad():
        for batch in data:

            x = batch['x']
            label = batch['s']
            lengths = batch['l']

            pred_label, all_scores = model.forward_RE(x, label, lengths, train=False)
            tensors.append(pred_label)
            tensors_score.append(all_scores)

            all_pred_label += list(flatten(pred_label, lengths))  # has been flattened
            all_true_label += list(flatten(label, lengths))

    acc, p, r, f = eval_seq_token(seq_label_pred=all_pred_label, seq_label_true=all_true_label, o_idx=o_idx)
    acc_ner, p_ner, r_ner, f_ner, class_res = get_ner_fmeasure(golden_lists=all_true_label,
                                                               predict_lists=all_pred_label,
                                                               label_type="BIO", i2s=i2s, all_class=True)

    results = {
        'token-level': [acc, p, r, f],
        'entity-level': [acc_ner, p_ner, r_ner, f_ner, class_res]
    }
    print(results)
    pred_label_all = torch.cat(tensors, dim=0) # B x L
    pred_score_all = torch.cat(tensors_score, dim=0) # B x L x C
    pred_score_all[pred_score_all==0.99] = 1.0

    model.train()
    return pred_label_all, pred_score_all


def assign_automata(args_bak):
    if args_bak.dataset == 'ATIS-BIO':
        args_bak.automata_path = '../data/ATIS-BIO/automata/automata.INTEGRATE.1025152019-1603639219.365244.150.100:0.3857,150:0.3004,.seed|168|.random.3best.71states.1splits.pkl'
    elif args_bak.dataset == 'ATIS-ZH-BIO':
        args_bak.automata_path = '../data/ATIS-ZH-BIO/automata/IIID.automata.0308133803-1615210683.6103313.svd.2best.104states.random.3splits.100-0.1496150-0.0934200-0.0541.bio.rules.v1.config.pkl'
    elif args_bak.dataset == 'SNIPS-BIO':
        args_bak.automata_path = '../data/SNIPS-BIO/automata/IIID.automata.0323152125-1616512885.4562736.svd.2best.104states.random.1splits.200-0.0025250-0.0026300-0.0022.bio.rules.v1.config.pkl'
    else:
        raise NotImplementedError()

    return args_bak


def load_re_results(args):
    re_path = args.automata_path + '.re.score'

    if os.path.exists(re_path):
        return True, pickle.load(open(re_path, 'rb'))

    else:
        return False, None


def predict_by_RE(args):
    # Settings
    dataset_name = args.dataset
    seq_max_len = args.seq_max_len
    bz = args.bz
    seed = args.seed
    args_bak = copy(args)

    # change the arguments for RE
    args_bak.data_type = 'all'
    args_bak.beta = 1
    args_bak.threshold = 0.99
    args_bak.rand_constant = 0
    args_bak.use_crf = 0

    args_bak = assign_automata(args_bak)

    set_seed(seed)

    saved, re = load_re_results(args_bak)
    if saved:
        return re

    dset = load_slot_dataset(dataset_name)
    t2i, i2t, s2i, i2s = dset['t2i'], dset['i2t'], dset['s2i'], dset['i2s']
    query_train, slot_train = dset['query_train'], dset['intent_train']
    query_dev, slot_dev = dset['query_dev'], dset['intent_dev']
    query_test, slot_test = dset['query_test'], dset['intent_test']

    len_stats(query_train)
    len_stats(query_dev)
    len_stats(query_test)
    # extend the padding
    # add pad <pad> to the last of vocab
    i2t[len(i2t)] = '<pad>'
    t2i['<pad>'] = len(i2t) - 1

    train_query, _, train_lengths = pad_dataset_1(query_train, seq_max_len, t2i['<pad>'])
    dev_query, _, dev_lengths = pad_dataset_1(query_dev, seq_max_len, t2i['<pad>'])
    test_query, _, test_lengths = pad_dataset_1(query_test, seq_max_len, t2i['<pad>'])
    slot_train, _, _ = pad_dataset_1(slot_train, seq_max_len, s2i['o'])
    slot_dev, _, _ = pad_dataset_1(slot_dev, seq_max_len, s2i['o'])
    slot_test, _, _ = pad_dataset_1(slot_test, seq_max_len, s2i['o'])

    slot_data_train = SlotBatchDatasetNoRE(train_query, train_lengths, slot_train, args_bak, s2i)
    slot_data_dev = SlotBatchDatasetNoRE(dev_query, dev_lengths, slot_dev, args_bak, s2i)
    slot_data_test = SlotBatchDatasetNoRE(test_query, test_lengths, slot_test, args_bak, s2i)
    print('Train Samples: ', len(slot_data_train))

    slot_dataloader_train = DataLoader(slot_data_train, batch_size=bz)
    slot_dataloader_dev = DataLoader(slot_data_dev, batch_size=bz)
    slot_dataloader_test = DataLoader(slot_data_test, batch_size=bz)

    priority_mat = create_mat_priority_MITR(s2i)

    automata = load_pkl(args_bak.automata_path)
    if 'automata' in automata:
        automata = automata['automata']

    print("AUTOMATA STATES NUM: {}".format(len(automata['states'])))

    if args.independent == 1:

        language_tensor, state2idx, wildcard_mat, output_tensor, output_wildcard_mat, final_vector, start_vector, language = \
            dfa_to_tensor_slot_independent(automata, t2i, s2i)

        model = FARNN_S_O_I(language_tensor,
                            output_tensor,
                            wildcard_mat,
                            output_wildcard_mat,
                            final_vector,
                            start_vector,
                            priority_mat,
                            args_bak,
                            o_idx=s2i['o'])

    elif args.independent == 2:

        if args.local_loss_func == 'CE1': func = dfa_to_tensor_slot_single_wildcard
        else: func = dfa_to_tensor_slot_single

        language_tensor, state2idx, wildcard_mat, output_mat, output_wildcard_vector, final_vector, start_vector, language = \
            func(automata, t2i, s2i)

        model = FARNN_S_O_I_S(language_tensor,
                            output_mat,
                            wildcard_mat,
                            output_wildcard_vector,
                            final_vector,
                            start_vector,
                            priority_mat,
                            args_bak,
                            o_idx=s2i['o'])

    else:
        language_tensor, state2idx, wildcard_tensor, wildcard_wildcard_tensor, final_vector, start_vector, language = \
            dfa_to_tensor_slot_new(automata, t2i, s2i)

        model = FARNN_S_O(language_tensor,
                          wildcard_tensor,
                          wildcard_wildcard_tensor,
                          final_vector,
                          start_vector,
                          priority_mat,
                          args_bak,
                          o_idx=s2i['o'])

    results_train, score_train = get_RE_prediction(slot_dataloader_train, model, args_bak, s2i['o'], i2s) # N x L x C

    results_dev, score_dev = get_RE_prediction(slot_dataloader_dev, model, args_bak, s2i['o'], i2s)

    results_test, score_test = get_RE_prediction(slot_dataloader_test, model, args_bak, s2i['o'], i2s)

    pickle.dump((results_train, results_dev, results_test, score_train, score_dev, score_test), open(args_bak.automata_path + '.re.score', 'wb'))

    return results_train, results_dev, results_test, score_train, score_dev, score_test


