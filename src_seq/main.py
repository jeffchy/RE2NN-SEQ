import sys
sys.path.append('../../')
sys.path.append('../')

from src_seq.train_onehot import train_slot_onehot
from src_seq.train_baseline import train_slot_neural_softmax_and_marryup_baselines
from src_seq.train_baseline_ptm import train_slot_neural_softmax_and_marryup_baselines_ptm

import argparse
from src_seq.train_decompose import train_slot_decompose
from src_seq.train_decompose_ptm import train_slot_decompose_ptm
import pickle

def parse_args():

    parser = argparse.ArgumentParser()

    ############## Arguments
    parser.add_argument('--dataset', type=str, default='SNIPS-BIO', help="dataset dir")
    parser.add_argument('--seq_max_len', type=int, default=30, help="Max seq length")
    parser.add_argument('--bz', type=int, default=500, help="batch size")
    parser.add_argument('--embed_dim', type=int, default=100, help="embed dim")
    parser.add_argument('--embed_type', type=str, default='glove', help='embedding type should be in [glove, fasttext]')
    parser.add_argument('--epoch', type=int, default=20, help="max state of each FSARNN")
    parser.add_argument('--train_portion', type=float, default=1.0, help="train portion")
    parser.add_argument('--automata_path', type=str, default="../data/MITR-toy/automata/automata.dict", help="automata path")
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument('--run', type=str, default='test', help="run string")
    parser.add_argument('--random_embed', type=int, default=0, help="0 false 1 true")
    parser.add_argument('--optimizer', type=str, default='ADAM', help="optimizer")
    parser.add_argument('--lr', type=float, default=0.0001, help="learning rate of optimizer")
    parser.add_argument('--train_mode', type=str, default='sum', help='global train mode, should be in [max, sum]')
    parser.add_argument('--local_loss_func', type=str, default='CE1', help='loss function in local mode, should be in [CE, NNLL, HL]')
    parser.add_argument('--rand_constant', type=float, default=1e-5, help='random noise')
    parser.add_argument('--threshold', type=float, default=0.5, help='the threshold used in the hinge loss option in local training')
    parser.add_argument('--margin', type=float, default=0.3, help='the margin used in the hinge loss option in local training')

    parser.add_argument('--select_level', type=str, default='entity-level', help='should be in entity-level or token-level')
    parser.add_argument('--method', type=str, default='onehot', help='method should be in [onehot, decompose, baseline]')
    parser.add_argument('--data_type', type=str, default='all', help='data type we use, should be in [all, re, n_re]')

    ############## Arguments for baselines
    parser.add_argument('--train_word_embed', type=int, default=0, help='if we train word embed or not, fix as default (0)')
    parser.add_argument('--rnn_hidden_dim', type=int, default=100, help='rnn / farnn_random hidden dim')
    parser.add_argument('--rnn', type=str, default='RNN', help='should be in RNN, LSTM, GRU')
    parser.add_argument('--bidirection', type=int, default=0, help='1 means bidirectional, 0 otherwise')
    parser.add_argument('--marryup_type', type=str, default='none', help='marryup type, [input, output, all, kd, pr]')
    parser.add_argument('--re_tag_dim', type=int, default=20, help='re tag embedding dim for marryup methods')
    parser.add_argument('--c1_kdpr', type=float, default=1, help='regularization param for PR, the bigger the harder, or the temperature in KD')
    parser.add_argument('--c2_kdpr', type=float, default=1, help='balancing weights for KD PR loss and original loss, [0, 1], 1 means use original loss (CE)')
    parser.add_argument('--c3_pr', type=float, default=1, help='annealing speed for pr')


    ############## Arguments for decompsed
    parser.add_argument('--normalize_automata', type=str, default='l2-rank', help='if we normalize the decomposed automata params, only used when use decomposed method [none, l1, l2]')
    parser.add_argument('--train_V_embed', type=int, default=0, help='0 means do not train V_embed')
    parser.add_argument('--beta', type=float, default=1.0, help='interpolation weight for word embedding and rule embedding')
    parser.add_argument('--rank', type=int, default=150, help='rank of decomposed tensor')
    parser.add_argument('--rank_wildcard', type=int, default=50, help='rank of wildcard decomposed tensor, should be in [30, 50, 70, 100, 150]')

    parser.add_argument('--additional_nonlinear', type=str, default='none', help='additional nonlinear for word embedding to rule dim')
    parser.add_argument('--additional_states', type=int, default=0, help='additional states with very small random values')
    parser.add_argument('--use_priority', type=int, default=0, help='0, or 1, 1 means use priority')
    parser.add_argument('--train_wildcard', type=int, default=0, help='if we train wildcard tensor CxSxS')
    parser.add_argument('--train_wildcard_wildcard', type=int, default=0, help='if we train wildcard_wildcard matrix SxS')
    parser.add_argument('--train_c_output', type=int, default=1, help='if we train C related params in single')
    parser.add_argument('--train_h0', type=int, default=0, help='if we train h0')
    parser.add_argument('--train_hT', type=int, default=0, help='if we train hT')
    parser.add_argument('--train_beta', type=int, default=0, help='if we train beta')
    parser.add_argument('--random', type=int, default=0, help='if we use random initialization')
    parser.add_argument('--random_pad_func', type=str, default='uniform', help='which padding function we use, \
                        should be in normal, uniform, and xavier')
    parser.add_argument('--save_model', type=int, default=0, help='if we save model')
    parser.add_argument('--independent', type=int, default=0, help='if we use independent decomposed model')
    parser.add_argument('--use_unlabel', type=int, default=0, help='if we use unlabel data')

    ############# Arguments for FAGRU
    parser.add_argument('--farnn', type=int, default=0,
                        help='0 for rnn, 1 for only update, 2 for update + reset')
    parser.add_argument('--xavier', type=int, default=0,
                        help='0 for not using xavier for farnn, only work when farnn > 1')
    parser.add_argument('--bias_init', type=float, default=5,
                        help='0 for not using xavier for farnn, only work when farnn > 1')
    parser.add_argument('--sigmoid_exponent', type=int, default=5,
                        help='sigmoidal function exponent')
    parser.add_argument('--use_crf', type=int, default=0, help='if we use crf')
    parser.add_argument('--update_nonlinear', type=str, default='none', help='whe nonlinear used when update')

    ############# For Save and Load
    parser.add_argument('--args_path', type=str, default='none', help='arguments path, if is not none, load and train')

    ############# For BERT
    parser.add_argument('--bert_finetune', type=int, default=0, help='if we finetune bert')
    parser.add_argument('--use_bert', type=int, default=0, help='if we use bert')
    parser.add_argument('--warm_up', type=int, default=0, help='if we use warm up')
    parser.add_argument('--bert_lr_down_factor', type=float, default=1,
                        help='the down factor for the (bert_lr = lr/down_factor)')
    parser.add_argument('--bert_init_embed', type=str, default='aggregate', help='embed used to initializing G, [random, aggregate, decontext]')

    return parser.parse_args(), parser


if __name__ == '__main__':


    args, parser = parse_args()

    if args.args_path != 'none':
        loaded_args = pickle.load(open(args.args_path, 'rb'))['args'].__dict__
        default_args = args.__dict__
        for k, v in loaded_args.items():
            if k not in default_args:
                print(k)


        for k, v in default_args.items():
            if k in loaded_args:
                default_args[k] = loaded_args[k]
            else:
                print(k)
        print(default_args)
        args = argparse.Namespace(**default_args)
        args.run = 'final_222'

    # Sanity check
    assert args.train_mode in ['max', 'sum']
    assert args.local_loss_func in ['CE1']
    assert args.update_nonlinear in ['none', 'relu', 'tanh', 'relutanh']

    assert args.rnn in ['LSTM', 'RNN', 'GRU']
    assert args.method in ['decompose', 'onehot', 'baseline']
    assert args.normalize_automata in ['none', 'l1', 'l2', 'l1-rank', 'l2-rank']
    assert args.additional_nonlinear in ['none', 'relu', 'tanh', 'sigmoid', 'relutanh']
    assert args.select_level in ['entity-level', 'token-level']

    assert args.rank in [30, 100, 150, 200, 250, 300, 350]
    assert args.rank_wildcard in [20, 30, 50, 70, 100, 150]
    assert args.random_pad_func in ['normal', 'xavier', 'uniform']
    assert args.seed in [0, 1, 2, 3, 4, 5]
    assert args.data_type in ['all', 're', 'n_re']
    # assert args.train_portion in [0, 0.003, 0.01, 0.1, 1.0, 20.0, 10.0, 50]
    assert args.independent in [0, 1, 2]

    if args.bert_finetune == 1:
        assert args.bert_lr_down_factor >= 5

    if args.train_portion == 0:
        assert args.epoch == 0

    if args.normalize_automata != 'none':
        assert args.method == 'decompose'

    if args.select_level == 'entity-level':
        assert 'BIO' in args.dataset

    if args.use_crf == 1:
        assert args.local_loss_func in ['CE', 'CE1']

    if args.random == 1:
        assert args.method != 'baseline'

    if args.method == 'decompose':
        assert args.marryup_type in ['none', 'kd', 'pr']
        # if args.farnn == 0:
            # assert args.bias_init == parser.get_default('bias_init')

    if args.method == 'baseline':
        assert args.marryup_type in ['none', 'input', 'output', 'all', 'pr', 'kd']
        if args.marryup_type == 'kd':
            assert args.c3_pr == parser.get_default('c3_pr')
            assert args.c1_kdpr >= 1.0
        elif args.marryup_type == 'pr':
            assert args.c1_kdpr >= 1.0

    if args.method == 'onehot':
        assert args.rand_constant == 0

    assert args.embed_type in ['glove', 'fasttext']
    assert args.dataset in ['ATIS-BIO', 'ATIS-ZH-BIO', 'SNIPS-BIO']
    if args.dataset == 'ATIS-ZH-BIO':
        assert args.embed_type == 'fasttext'

    if not bool(args.use_bert):
        assert args.warm_up == 0
        assert args.bert_finetune == 0
        assert args.bert_lr_down_factor == 1

    if args.method == 'onehot':
        train_slot_onehot(args)
    elif args.method == 'decompose':
        if args.use_bert:
            train_slot_decompose_ptm(args)
        else:
            train_slot_decompose(args)
    else:
        if args.use_bert:
            train_slot_neural_softmax_and_marryup_baselines_ptm(args)
        else:
            train_slot_neural_softmax_and_marryup_baselines(args)
