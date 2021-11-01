import torch
from src_seq.utils import Logger, len_stats, pad_dataset_1, set_seed
from src_seq.data import load_slot_dataset, SlotBatchDataset
from src_seq.val import val_onehot_verbose, val_baselines_verbose
from src_seq.tools.printer import Best_Model_Recorder, print_and_log_results
from src_seq.farnn.model_decompose import FARNN_S_D_W
from src_seq.farnn.model_decompose_independent import FARNN_S_D_W_I
from src_seq.farnn.model_decompose_single import FARNN_S_D_W_I_S
from src_seq.init_params import get_init_params_seq_independent, get_init_params_seq, \
                                get_init_params_seq_independent_single, get_init_random_params
from src_seq.data import load_slot_dataset, MarryUpSlotBatchDataset
from torch.utils.data import DataLoader
from src_seq.data import load_glove_embed
from src_seq.baselines.neural_softmax import SlotNeuralSoftmax
import argparse
import pickle
import numpy as np
from src_seq.tools.saver import save_model_and_log


def test_slot_decompose(args, res):

    # Settings
    logger = Logger()
    dataset_name = args.dataset
    seq_max_len = args.seq_max_len
    train_portion = args.train_portion
    bz = args.bz
    optimizer = args.optimizer
    seed = args.seed

    set_seed(seed)

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

    slot_data_train = SlotBatchDataset(train_query, train_lengths, slot_train, args, s2i,
                                       portion=train_portion, dset='train',)
    slot_data_dev = SlotBatchDataset(dev_query, dev_lengths, slot_dev, args, s2i,
                                     portion=train_portion, dset='dev',)
    slot_data_test = SlotBatchDataset(test_query, test_lengths, slot_test,
                                      args, s2i, dset='test',)

    print('Train Samples: ', len(slot_data_train))
    print('Dev Samples: ', len(slot_data_dev))
    print('Test Samples: ', len(slot_data_test))

    slot_dataloader_train = DataLoader(slot_data_train, batch_size=bz)
    slot_dataloader_dev = DataLoader(slot_data_dev, batch_size=bz)
    slot_dataloader_test = DataLoader(slot_data_test, batch_size=bz)

    vocab_size = len(t2i)
    label_size = len(s2i)

    # TODO: we currently not support CE1 for 4-order method
    if args.local_loss_func == 'CE1':
       assert args.independent != 0

    if args.independent == 0:  # 4-th order tensor model
        V_embed_extend, C_embed, S1, S2, pretrain_embed_extend, wildcard_tensor, wildcard_wildcard_tensor, final_vector, start_vector, priority_mat, \
            C_wildcard, S1_wildcard, S2_wildcard = get_init_params_seq(args, s2i)

        model = FARNN_S_D_W(
            V=V_embed_extend,
            C=C_embed,
            S1=S1,
            S2=S2,
            C_wildcard=C_wildcard,
            S1_wildcard=S1_wildcard,
            S2_wildcard=S2_wildcard,
            wildcard_wildcard=wildcard_wildcard_tensor,
            final_vector=final_vector,
            start_vector=start_vector,
            pretrained_word_embed=pretrain_embed_extend,
            priority_mat=priority_mat,
            args=args,
            o_idx=s2i['o'])

    elif args.independent == 2:  # single model, 3-order tensor + matrix

        V_embed_extend, S1, S2, pretrain_embed_extend, wildcard_mat, wildcard_output_vector, \
        final_vector, start_vector, priority_mat, C_output_mat, _ \
        = get_init_params_seq_independent_single(args, s2i, t2i)

        model = FARNN_S_D_W_I_S(
            V=V_embed_extend,
            S1=S1,
            S2=S2,
            C_output_mat=C_output_mat,
            wildcard_mat=wildcard_mat,
            wildcard_output_vector=wildcard_output_vector,
            final_vector=final_vector,
            start_vector=start_vector,
            pretrained_word_embed=pretrain_embed_extend,
            priority_mat=priority_mat,
            args=args,
            o_idx=s2i['o'])

    else: # 2 3-order tensors
        V_embed_extend, S1, S2, pretrain_embed_extend, wildcard_mat, wildcard_output, \
        final_vector, start_vector, priority_mat, C_output, S1_output, S2_output \
        = get_init_params_seq_independent(args, s2i, t2i)

        model = FARNN_S_D_W_I(
            V=V_embed_extend,
            S1=S1,
            S2=S2,
            C_output=C_output,
            S1_output=S1_output,
            S2_output=S2_output,
            wildcard_mat=wildcard_mat,
            wildcard_output=wildcard_output,
            final_vector=final_vector,
            start_vector=start_vector,
            pretrained_word_embed=pretrain_embed_extend,
            priority_mat=priority_mat,
            args=args,
            o_idx=s2i['o'])

    model.load_state_dict(res.best_model_state_dict)

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=0)
    if args.optimizer == 'ADAM':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)

    if torch.cuda.is_available():
        model = model.cuda()

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('ALL TRAINABLE PARAMETERS: {}'.format(pytorch_total_params))

    results_train = val_onehot_verbose(slot_dataloader_train, model, args, s2i['o'], i2s)
    print_and_log_results(logger, results_train, 'INIT', 'TRAIN')

    results_dev = val_onehot_verbose(slot_dataloader_dev, model, args, s2i['o'], i2s)
    print_and_log_results(logger, results_dev, 'INIT', 'DEV')

    results_test = val_onehot_verbose(slot_dataloader_test, model, args, s2i['o'], i2s)
    print_and_log_results(logger, results_test, 'INIT', 'TEST')

    save_model_and_log(logger, [results_train, results_dev, results_test], args)


def test_slot_neural_softmax_and_marryup_baselines(args, res):

    # Settings
    logger = Logger()
    dataset_name = args.dataset
    seq_max_len = args.seq_max_len
    train_portion = args.train_portion
    bz = args.bz
    random_embed = bool(args.random_embed)
    optimizer = args.optimizer
    seed = args.seed
    embed_dim = args.embed_dim

    set_seed(seed)

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

    slot_data_train = MarryUpSlotBatchDataset(train_query, train_lengths, slot_train,
                                              args, s2i, train_portion, dset='train')
    slot_data_dev = MarryUpSlotBatchDataset(dev_query, dev_lengths, slot_dev,
                                            args, s2i, train_portion, dset='dev')
    slot_data_test = MarryUpSlotBatchDataset(test_query, test_lengths, slot_test,
                                             args, s2i, dset='test')

    print('Train Samples: ', len(slot_data_train))
    print('Dev Samples: ', len(slot_data_dev))
    print('Test Samples: ', len(slot_data_test))

    slot_dataloader_train = DataLoader(slot_data_train, batch_size=bz)
    slot_dataloader_dev = DataLoader(slot_data_dev, batch_size=bz)
    slot_dataloader_test = DataLoader(slot_data_test, batch_size=bz)

    label_size = len(s2i)

    pretrained_embed = load_glove_embed('../data/{}/'.format(dataset_name), embed_dim)
    if random_embed: pretrained_embed = np.random.random(pretrained_embed.shape)
    pretrained_embed = np.append(pretrained_embed, np.zeros((1, args.embed_dim), dtype=np.float), axis=0)

    # for padding
    model = SlotNeuralSoftmax(pretrained_embed=pretrained_embed,
                              args=args,
                              label_size=label_size,
                              o_idx=s2i['o'])

    model.load_state_dict(res.best_model_state_dict)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('ALL TRAINABLE PARAMETERS: {}'.format(pytorch_total_params))

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=0)
    if args.optimizer == 'ADAM':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)

    if torch.cuda.is_available():
        model = model.cuda()

    results_train = val_baselines_verbose(slot_dataloader_train, model, args, s2i['o'], i2s)
    print_and_log_results(logger, results_train, 'INIT', 'TRAIN')

    results_dev = val_baselines_verbose(slot_dataloader_dev, model, args, s2i['o'], i2s)
    print_and_log_results(logger, results_dev, 'INIT', 'DEV')

    results_test = val_baselines_verbose(slot_dataloader_test, model, args, s2i['o'], i2s)
    print_and_log_results(logger, results_test, 'INIT', 'TEST')

    save_model_and_log(logger, [results_train, results_dev, results_test], args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_path', type=str, default='none')
    args = parser.parse_args()

    all = pickle.load(open(args.res_path, 'rb'))
    args = all['args']
    res = all['res']
    if res.best_model_state_dict is None:
        print('SAVED MODEL IS NONE!')
        exit(0)

    if args.method == 'decompose':
        test_slot_decompose(args, res)
    else:
        test_slot_neural_softmax_and_marryup_baselines(args, res)


