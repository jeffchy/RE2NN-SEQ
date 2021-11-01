import numpy as np
from src_seq.data import load_glove_embed, load_fasttext_embed
from src_seq.utils import load_pkl, xavier_normal
from src_seq.utils import get_average
from src_seq.ptm.bert_utils import load_bert_embed
from src_seq.create_logic_mat_bias import create_mat_priority_MITR, \
    create_mat_priority_MITM, create_mat_priority_ATIS, \
    create_mat_priority_SNIPS, create_mat_priority



def get_init_params_seq(args, s2i, data_dir='../data/'):


    print("Start getting initial decompsoed parameters V C S1 S2...")
    dset = args.dataset
    if args.embed_type == 'glove':
        pretrained_embed = load_glove_embed('{}{}/'.format(data_dir, dset), args.embed_dim)
    else:
        pretrained_embed = load_fasttext_embed('{}{}/'.format(data_dir, dset), args.embed_dim)

    if args.random_embed: pretrained_embed = np.random.random(pretrained_embed.shape)

    automata_dicts = load_pkl(args.automata_path)
    print("Loading automata: {}".format(args.automata_path))
    automata_factor_dicts = automata_dicts[args.seed][0]
    automata_wildcard_factor_dicts = automata_dicts[args.seed][1]
    # TODO: we currently do not support CE1 for decompose

    factor_dicts = automata_factor_dicts[args.rank]
    factor_wildcard_dicts = automata_wildcard_factor_dicts[args.rank_wildcard]

    automata = automata_dicts['automata']
    V_embed, C_embed, S1, S2 = factor_dicts['V'], factor_dicts['C'], factor_dicts['S1'], factor_dicts['S2']
    wildcard_tensor = factor_dicts['wildcard_tensor'] # C x S x S
    wildcard_wildcard_tensor = factor_dicts['wildcard_wildcard_tensor']
    # the parameter for wildcard tensor splits
    C_wildcard, S1_wildcard, S2_wildcard = \
        factor_wildcard_dicts['C_wildcard'], \
        factor_wildcard_dicts['S1_wildcard'], \
        factor_wildcard_dicts['S2_wildcard']

    C , _ = C_embed.shape
    if args.local_loss_func == 'CE1': # sanity check
        assert C == len(s2i) + 1

    print("Clipping corrupted values after decomposition")
    corrupt_thres = 100

    invalid_positive = np.sum(V_embed > corrupt_thres) + np.sum(C_embed > corrupt_thres) + \
                       np.sum(S1 > corrupt_thres) + np.sum(S2 > corrupt_thres)
    V_embed[V_embed > corrupt_thres] = 1
    C_embed[C_embed > corrupt_thres] = 1
    S1[S1 > corrupt_thres] = 1
    S2[S2 > corrupt_thres] = 1
    print('Invalid Positive Values: {}'.format(invalid_positive))

    invalid_negative = np.sum(V_embed < -corrupt_thres) + np.sum(C_embed < -corrupt_thres) + \
                       np.sum(S1 < -corrupt_thres) + np.sum(S2 < -corrupt_thres)
    V_embed[V_embed < -corrupt_thres] = -1
    C_embed[C_embed < -corrupt_thres] = -1
    S1[S1 < -corrupt_thres] = -1
    S2[S2 < -corrupt_thres] = -1
    print('Invalid Negative Values: {}'.format(invalid_negative))

    max_states, rank = S1.shape

    final_vector = np.zeros(max_states)
    final_vector[automata['finalstates']] = 1

    start_vector = np.zeros(max_states)
    start_vector[automata['startstate']] = 1

    n_vocab, rank = V_embed.shape
    n_state, _ = S1.shape
    print("DFA states: {}".format(n_state))
    _, embed_dim = pretrained_embed.shape

    # for padding
    pretrain_embed_extend = np.append(pretrained_embed, np.zeros((1, args.embed_dim), dtype=np.float), axis=0)
    V_embed_extend = np.append(V_embed, np.zeros((1, rank), dtype=np.float), axis=0)

    priority_mat = create_mat_priority(s2i, args)

    if args.normalize_automata != 'none':
        print('Normalize automata decomposed parameters...')
        C_embed_avg = get_average(C_embed, args.normalize_automata)
        S1_avg = get_average(S1, args.normalize_automata)
        S2_avg = get_average(S2, args.normalize_automata)
        V_embed_extend_avg = get_average(V_embed_extend, args.normalize_automata)
        factor = np.float_power(C_embed_avg*S1_avg*S2_avg*V_embed_extend_avg, 1/4)

        print(factor)
        print("S1 avg norm: ", S1_avg)
        print("S2 avg norm: ", S2_avg)
        print("C_embed avg norm: ", C_embed_avg)
        print("V_embed_extend avg norm: ", V_embed_extend_avg)

        S1 = S1 * (factor / S1_avg)
        S2 = S2 * (factor / S2_avg)
        C_embed = C_embed * (factor / C_embed_avg)
        V_embed_extend = V_embed_extend * (factor / V_embed_extend_avg)

        if C_wildcard is not None:
            C_embed_avg_wildcard = get_average(C_wildcard, args.normalize_automata)
            S1_avg_wildcard = get_average(S1_wildcard, args.normalize_automata)
            S2_avg_wildcard = get_average(S2_wildcard, args.normalize_automata)
            factor = np.float_power(C_embed_avg_wildcard * S1_avg_wildcard * S2_avg_wildcard, 1/3)

            print(factor)
            print("S1_wildcard avg norm: ", S1_wildcard)
            print("S2_wildcard avg norm: ", S2_wildcard)
            print("C_embed avg norm: ", C_embed_avg_wildcard)

            S1_wildcard = S1_wildcard * (factor / S1_avg_wildcard)
            S2_wildcard = S2_wildcard * (factor / S2_avg_wildcard)
            C_wildcard = C_wildcard * (factor / C_embed_avg_wildcard)

    return V_embed_extend, C_embed, S1, S2, pretrain_embed_extend, wildcard_tensor, wildcard_wildcard_tensor, \
           final_vector, start_vector, priority_mat, C_wildcard, S1_wildcard, S2_wildcard


def get_init_params_seq_independent(args, s2i, t2i, data_dir='../data/'):
    print("Start getting initial decompsoed parameters V C S1 S2...")

    dset = args.dataset

    if args.embed_type == 'glove':
        pretrained_embed = load_glove_embed('{}{}/'.format(data_dir, dset), args.embed_dim)
    else:
        pretrained_embed = load_fasttext_embed('{}{}/'.format(data_dir, dset), args.embed_dim)

    if args.random_embed: pretrained_embed = np.random.random(pretrained_embed.shape)

    automata_dicts = load_pkl(args.automata_path)
    print("Loading automata: {}".format(args.automata_path))
    automata_factor_dicts = automata_dicts[args.seed][0]
    if args.local_loss_func == 'CE1':
        automata_output_factor_dicts = automata_dicts[args.seed][2]
    else:
        automata_output_factor_dicts = automata_dicts[args.seed][1]

    factor_dicts = automata_factor_dicts[args.rank]
    factor_output_dicts = automata_output_factor_dicts[args.rank_wildcard]
    automata = automata_dicts['automata']

    V_embed, S1, S2, wildcard_mat = \
        factor_dicts['V'], factor_dicts['S1'], factor_dicts['S2'], factor_dicts['wildcard_mat']

    C_output, S1_output, S2_output, wildcard_output = factor_output_dicts['C_output'],\
                                                      factor_output_dicts['S1_output'], \
                                                      factor_output_dicts['S2_output'], \
                                                      factor_output_dicts['wildcard_output']

    print("Clipping corrupted values after decomposition")
    corrupt_thres = 1000

    invalid_positive = np.sum(V_embed > corrupt_thres) + \
                       np.sum(S1 > corrupt_thres) + np.sum(S2 > corrupt_thres)
    # V_embed[V_embed > corrupt_thres] = 1
    # S1[S1 > corrupt_thres] = 1
    # S2[S2 > corrupt_thres] = 1
    print('Invalid Positive Values: {}'.format(invalid_positive))

    invalid_negative = np.sum(V_embed < -corrupt_thres) + \
                       np.sum(S1 < -corrupt_thres) + np.sum(S2 < -corrupt_thres)
    # V_embed[V_embed < -corrupt_thres] = -1
    # S1[S1 < -corrupt_thres] = -1
    # S2[S2 < -corrupt_thres] = -1
    print('Invalid Negative Values: {}'.format(invalid_negative))

    max_states, rank = S1.shape

    final_vector = np.zeros(max_states)
    final_vector[automata['finalstates']] = 1

    start_vector = np.zeros(max_states)
    start_vector[automata['startstate']] = 1

    n_vocab, rank = V_embed.shape
    n_state, _ = S1.shape
    print("DFA states: {}".format(n_state))
    _, embed_dim = pretrained_embed.shape

    # for padding
    pretrain_embed_extend = np.append(pretrained_embed, np.zeros((1, args.embed_dim), dtype=np.float), axis=0)
    V_embed_extend = np.append(V_embed, np.zeros((1, rank), dtype=np.float), axis=0)

    priority_mat = create_mat_priority(s2i, args)

    if args.normalize_automata != 'none':
        print('Normalize automata decomposed parameters...')
        V_embed_extend_avg = get_average(V_embed_extend, args.normalize_automata)
        S1_avg = get_average(S1, args.normalize_automata)
        S2_avg = get_average(S2, args.normalize_automata)
        factor = np.float_power(V_embed_extend_avg*S1_avg*S2_avg, 1/3)
        print(factor)
        print("V_embed_extend avg norm: ", V_embed_extend_avg)
        print("S1 avg norm: ", S1_avg)
        print("S2 avg norm: ", S2_avg)
        S1 = S1 * (factor / S1_avg)
        S2 = S2 * (factor / S2_avg)
        V_embed_extend = V_embed_extend * (factor / V_embed_extend_avg)

        C_output_avg = get_average(C_output, args.normalize_automata)
        S1_output_avg = get_average(S1_output, args.normalize_automata)
        S2_output_avg = get_average(S2_output, args.normalize_automata)
        factor = np.float_power(C_output_avg*S1_output_avg*S2_output_avg, 1/3)
        print(factor)
        print("C_output avg norm: ", C_output_avg)
        print("S1_output avg norm: ", S1_output_avg)
        print("S2_output avg norm: ", S2_output_avg)
        C_output = C_output * (factor / C_output_avg)
        S1_output = S1_output * (factor / S1_output_avg)
        S2_output = S2_output * (factor / S2_output_avg)

    return V_embed_extend, S1, S2, pretrain_embed_extend, wildcard_mat, wildcard_output, \
           final_vector, start_vector, priority_mat, C_output, S1_output, S2_output


def get_init_params_seq_independent_single(args, s2i, t2i, data_dir='../data/'):
    print("Start getting initial decompsoed parameters V C S1 S2...")

    dset = args.dataset
    if args.embed_type == 'glove':
        pretrained_embed = load_glove_embed('{}{}/'.format(data_dir, dset), args.embed_dim)
    else:
        pretrained_embed = load_fasttext_embed('{}{}/'.format(data_dir, dset), args.embed_dim)
    if args.random_embed: pretrained_embed = np.random.random(pretrained_embed.shape)

    automata_dicts = load_pkl(args.automata_path)
    print("Loading automata: {}".format(args.automata_path))
    automata_factor_dicts = automata_dicts[args.seed][0]
    if args.local_loss_func == 'CE1':
        automata_output_factor_dicts = automata_dicts[args.seed][2]
    else:
        automata_output_factor_dicts = automata_dicts[args.seed][1]

    factor_dicts = automata_factor_dicts[args.rank]
    factor_output_dicts = automata_output_factor_dicts
    automata = automata_dicts['automata']

    V_embed, S1, S2, wildcard_mat = \
        factor_dicts['V'], factor_dicts['S1'], factor_dicts['S2'], factor_dicts['wildcard_mat']

    C_output_mat, wildcard_output_vector = \
        factor_output_dicts['output_mat'], factor_output_dicts['output_wildcard_vector']

    print("Clipping corrupted values after decomposition")
    corrupt_thres = 1e5

    invalid_positive = np.sum(V_embed > corrupt_thres) + \
                       np.sum(S1 > corrupt_thres) + np.sum(S2 > corrupt_thres)
    # V_embed[V_embed > corrupt_thres] = 1
    # S1[S1 > corrupt_thres] = 1
    # S2[S2 > corrupt_thres] = 1
    print('Invalid Positive Values: {}'.format(invalid_positive))

    invalid_negative = np.sum(V_embed < -corrupt_thres) + \
                       np.sum(S1 < -corrupt_thres) + np.sum(S2 < -corrupt_thres)
    # V_embed[V_embed < -corrupt_thres] = -1
    # S1[S1 < -corrupt_thres] = -1
    # S2[S2 < -corrupt_thres] = -1
    print('Invalid Negative Values: {}'.format(invalid_negative))

    max_states, rank = S1.shape

    final_vector = np.zeros(max_states)
    final_vector[automata['finalstates']] = 1

    start_vector = np.zeros(max_states)
    start_vector[automata['startstate']] = 1

    n_vocab, rank = V_embed.shape
    n_state, _ = S1.shape
    print("DFA states: {}".format(n_state))
    _, embed_dim = pretrained_embed.shape

    # for padding
    pretrain_embed_extend = np.append(pretrained_embed, np.zeros((1, args.embed_dim), dtype=np.float), axis=0)
    V_embed_extend = np.append(V_embed, np.zeros((1, rank), dtype=np.float), axis=0)

    priority_mat = create_mat_priority(s2i, args)

    if args.normalize_automata != 'none':
        print('Normalize automata decomposed parameters...')
        V_embed_extend_avg = get_average(V_embed_extend, args.normalize_automata)
        S1_avg = get_average(S1, args.normalize_automata)
        S2_avg = get_average(S2, args.normalize_automata)
        factor = np.float_power(V_embed_extend_avg*S1_avg*S2_avg, 1/3)
        print(factor)
        print("V_embed_extend avg norm: ", V_embed_extend_avg)
        print("S1 avg norm: ", S1_avg)
        print("S2 avg norm: ", S2_avg)
        S1 = S1 * (factor / S1_avg)
        S2 = S2 * (factor / S2_avg)
        V_embed_extend = V_embed_extend * (factor / V_embed_extend_avg)

    if args.random == 1:
        V_embed_extend = xavier_normal(V_embed_extend)
        S1 = xavier_normal(S1)
        S2 = xavier_normal(S2)
        wildcard_mat = xavier_normal(wildcard_mat)
        wildcard_output_vector = xavier_normal(wildcard_output_vector)
        final_vector = xavier_normal(final_vector)
        start_vector = xavier_normal(start_vector)
        C_output_mat = xavier_normal(C_output_mat)
        assert args.use_priority == 0

    V, _ = V_embed_extend.shape
    bert_embed = None
    if bool(args.use_bert):
        if args.bert_init_embed == 'random':
            bert_embed = np.random.randn(V, 768)*0.5
        else:
            bert_embed = load_bert_embed(args)
            bert_embed = np.append(bert_embed, np.zeros((1, 768), dtype=np.float), axis=0)

    return V_embed_extend, S1, S2, pretrain_embed_extend, wildcard_mat, wildcard_output_vector, \
           final_vector, start_vector, priority_mat, C_output_mat, bert_embed


def get_init_random_params(args, s2i, t2i, data_dir='../data/'):
    dset = args.dataset
    rank = args.rank
    rank_wildcard = args.rank_wildcard

    n_vocab = len(t2i) - 1
    n_state = args.rnn_hidden_dim
    n_label = len(s2i)

    if args.random_embed: pretrained_embed = np.random.random((n_vocab, args.embed_dim))
    else:
        if args.embed_type == 'glove':
            pretrained_embed = load_glove_embed('{}{}/'.format(data_dir, dset), args.embed_dim)
        else:
            pretrained_embed = load_fasttext_embed('{}{}/'.format(data_dir, dset), args.embed_dim)

    print("DFA states: {}".format(n_state))
    _, embed_dim = pretrained_embed.shape

    V_embed = np.zeros((n_vocab, rank), dtype=np.float)
    # for padding
    pretrain_embed_extend = np.append(pretrained_embed, np.zeros((1, args.embed_dim), dtype=np.float), axis=0)
    V_embed_extend = np.append(V_embed, np.zeros((1, rank), dtype=np.float), axis=0)
    C_embed = np.zeros((n_label, rank), dtype=np.float)
    S1 = np.zeros((n_state, rank), dtype=np.float)
    S2 = np.zeros((n_state, rank), dtype=np.float)
    wildcard_tensor = np.zeros((n_label, n_state, n_state), dtype=np.float)
    wildcard_wildcard_tensor = np.zeros((n_state, n_state), dtype=np.float)
    final_vector = np.zeros(n_state, dtype=np.float)
    start_vector = np.zeros(n_state, dtype=np.float)
    C_wildcard = np.zeros((n_label, rank_wildcard), dtype=np.float)
    S1_wildcard = np.zeros((n_state, rank_wildcard), dtype=np.float)
    S2_wildcard = np.zeros((n_state, rank_wildcard), dtype=np.float)
    priority_mat = np.zeros((n_label, n_label), dtype=np.float)
    return V_embed_extend, C_embed, S1, S2, pretrain_embed_extend, wildcard_tensor, wildcard_wildcard_tensor, \
           final_vector, start_vector, priority_mat, C_wildcard, S1_wildcard, S2_wildcard