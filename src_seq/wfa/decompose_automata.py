from src_seq.load_data_and_rules import *
from src_seq.wfa.fsa_to_tensor import dfa_to_tensor_slot_new, dfa_to_tensor_slot_independent, \
    dfa_to_tensor_slot_single, dfa_to_tensor_slot_single_wildcard, dfa_to_tensor_slot_independent_wildcard, \
    dfa_to_tensor_slot_new_wildcard
from src_seq.wfa.tensor_func import decompose_tensor_split_slot_4d, decompose_tensor_split_slot_3d, \
    decompose_tensor_split_slot_3d_language
from src_seq.utils import create_datetime_str
from copy import copy, deepcopy
from src_seq.data import load_slot_dataset
from src_seq.utils import load_pkl
from src_seq.tools.timer import Timer
from numpy.linalg import LinAlgError

timer = Timer()


def decompose_automata(rearranged_automata,
                       dataset_name,
                       automata_name,
                       split_group,
                       init='random',
                       k_best=5,
                       rule_name=''):


    timer.start()
    print('AUTOMATA TO TENSOR')
    print('Total States: {}'.format(len(rearranged_automata['states'])))

    data = load_slot_dataset(dataset_name, '../../data/')
    # first load vocabs
    word2idx = data['t2i']
    slot2idx = data['s2i']

    language_tensor, state2idx, wildcard_tensor, wildcard_wildcard_tensor, final_vector, start_vector, language = \
        dfa_to_tensor_slot_new(rearranged_automata, word2idx, slot2idx, dataset=dataset_name)

    timer.stop()

    print('DECOMPOSE SPLIT AUTOMATA')
    if dataset_name == 'MITR':
        ranks = [100, 150, 200]
    elif dataset_name == 'MITR-BIO':
        ranks = [150, 200, 250]
    elif dataset_name == 'MITM-E-BIO':
        ranks = [150, 200, 250]
    elif dataset_name == 'ATIS-BIO':
        ranks = [100, 150, 200]
    elif dataset_name == 'ATIS-ZH-BIO':
        ranks = [150, 200, 250]
    elif dataset_name == 'CONLL03-BIO':
        ranks = [100, 150, 200]
    else:
        ranks = [100, 150, 200]

    time_str = create_datetime_str()
    n_states = len(rearranged_automata['states'])

    # We first split the wildcard_tensor sized C x S x S
    wildcard_tensor_ranks = [70, 100, 150]
    all_decomposed_automata = dict()
    all_decomposed_automata['automata'] = rearranged_automata

    print(language)
    for random_state in range(4):
        wildcard_factor_dicts = dict()
        for wildcard_tensor_rank in wildcard_tensor_ranks:
            print('DECOMPOSING WILDCARD TENSOR RANK: {}, TENSOR SIZE: {}'.format(wildcard_tensor_rank,
                                                                                 wildcard_tensor.shape))
            lowest_rec_error = float('inf')
            best_save_dict = None
            for k in range(k_best):
                print('BEST of {}: {}'.format(k_best, k))
                random_state = random_state + k * 8
                try:
                    C_embed_wildcard, S1_wildcard, S2_wildcard, rec_error = decompose_tensor_split_slot_3d(
                        wildcard_tensor, wildcard_tensor_rank, random_state=random_state,
                        init='random', verbose=1, n_iter_max=10)
                except ValueError:
                    continue

                if rec_error[-1] < lowest_rec_error:
                    wildcard_dict = {
                        'C_wildcard': C_embed_wildcard,
                        'S1_wildcard': S1_wildcard,
                        'S2_wildcard': S2_wildcard,
                    }
                    best_save_dict = copy(wildcard_dict)
                    lowest_rec_error = rec_error[-1]

            wildcard_factor_dicts[wildcard_tensor_rank] = best_save_dict

        timer.start()
        info_str = ''
        factor_dicts = dict()
        info_dicts = dict()
        for rank in ranks:
            print('DECOMPOSING RANK: {}, NON-NEGATIV: {}, TENSOR SIZE: {}'.format(rank, False, language_tensor.shape))

            lowest_rec_error = float('inf')
            best_save_dict = None
            for k in range(k_best):
                print('BEST of {}: {}'.format(k_best, k))
                random_state = random_state + k * 8
                try:
                    V_embed_split, C_embed_split, S1_split, S2_split, rec_error = decompose_tensor_split_slot_4d(
                        language_tensor, language, word2idx, rank, random_state=random_state,
                        init=init, verbose=1, n_iter_max=10)
                except (ValueError, LinAlgError):
                    print('Value Error or LinAlgError')
                    continue

                if rec_error[-1] < lowest_rec_error:
                    save_dict = {
                        'V': V_embed_split,
                        'C': C_embed_split,
                        'S1': S1_split,
                        'S2': S2_split,
                        'wildcard_tensor': wildcard_tensor,
                        'wildcard_wildcard_tensor': wildcard_wildcard_tensor
                    }
                    best_save_dict = deepcopy(save_dict)
                    lowest_rec_error = rec_error[-1]

            factor_dicts[rank] = best_save_dict
            info_dicts[rank] = lowest_rec_error.__round__(4)
            info_str += '{}:{},'.format(rank, info_dicts[rank])

        all_decomposed_automata[random_state] = [factor_dicts, wildcard_factor_dicts]
        timer.stop()

    pickle.dump(all_decomposed_automata,
                open(
                    '../../data/{dataset}/automata/D.automata.{time_str}.{init}.{k}best.{n}states.{automata_name}.{split}splits.{info}.{rule_name}.pkl'. \
                    format(dataset=dataset_name,
                           time_str=time_str,
                           init=init,
                           k=k_best,
                           n=n_states,
                           automata_name=automata_name,
                           split=split_group,
                           info=info_str,
                           rule_name=rule_name), 'wb')
                )
    print('FINISHED')


def decompose_automata_independent(rearranged_automata,
                                   dataset_name,
                                   automata_name,
                                   split_group,
                                   init='random',
                                   k_best=5,
                                   rule_name=''):

    timer.start()
    print('AUTOMATA TO TENSOR')
    print('Total States: {}'.format(len(rearranged_automata['states'])))

    data = load_slot_dataset(dataset_name, '../../data/')
    # first load vocabs
    word2idx = data['t2i']
    slot2idx = data['s2i']

    language_tensor, state2idx, wildcard_mat, output_tensor, output_wildcard_mat, final_vector, start_vector, language = \
        dfa_to_tensor_slot_independent(rearranged_automata, word2idx, slot2idx, dataset=dataset_name)

    _, _, _, new_output_tensor, _, _, _, _ = \
        dfa_to_tensor_slot_independent_wildcard(rearranged_automata, word2idx, slot2idx, dataset=dataset_name)

    timer.stop()

    print('DECOMPOSE SPLIT AUTOMATA')
    if dataset_name == 'MITR':
        ranks = [100, 150, 200]
    elif dataset_name == 'MITR-BIO':
        ranks = [150, 200, 250]
    elif dataset_name == 'MITM-E-BIO':
        ranks = [150, 200, 250]
    elif dataset_name == 'ATIS-BIO':
        ranks = [100, 150, 200]
    elif dataset_name == 'ATIS-ZH-BIO':
        ranks = [150, 200, 250]
    elif dataset_name == 'CONLL03-BIO':
        ranks = [100, 150, 200]
    else:
        ranks = [100, 150, 200]

    time_str = create_datetime_str()
    n_states = len(rearranged_automata['states'])

    # We first split the wildcard_tensor sized C x S x S
    output_tensor_ranks = [70, 100, 150]

    all_decomposed_automata = dict()
    all_decomposed_automata['automata'] = rearranged_automata

    for random_state in range(4):

        rands = random_state

        output_factor_dicts = dict() # C related params
        output_factor_dicts_w = dict() # C+1 related params
        for output_tensor_rank in output_tensor_ranks:
            print('DECOMPOSING WILDCARD TENSOR RANK: {}, TENSOR SIZE: {}'.format(output_tensor_rank,
                                                                                 output_tensor.shape))
            lowest_rec_error = float('inf')
            best_save_dict = None
            for k in range(k_best):
                print('BEST of {}: {}'.format(k_best, k))
                rands = rands + k * 8
                try:
                    C_embed_wildcard, S1_wildcard, S2_wildcard, rec_error = decompose_tensor_split_slot_3d(
                        output_tensor, output_tensor_rank, random_state=rands,
                        init=init, verbose=1, n_iter_max=40)
                except ValueError:
                    continue

                if rec_error[-1] < lowest_rec_error:
                    output_dict = {
                        'C_output': C_embed_wildcard,
                        'S1_output': S1_wildcard,
                        'S2_output': S2_wildcard,
                        'wildcard_output': output_wildcard_mat,
                    }
                    best_save_dict = copy(output_dict)
                    lowest_rec_error = rec_error[-1]

            output_factor_dicts[output_tensor_rank] = best_save_dict

            print('DECOMPOSING NEW WILDCARD TENSOR RANK: {}, TENSOR SIZE: {}'.format(output_tensor_rank,
                                                                                 new_output_tensor.shape))
            lowest_rec_error = float('inf')
            best_save_dict = None
            for k in range(k_best):
                print('BEST of {}: {}'.format(k_best, k))
                rands = rands + k * 8
                try:
                    C_embed_wildcard_new, S1_wildcard_new, S2_wildcard_new, rec_error = decompose_tensor_split_slot_3d(
                        new_output_tensor, output_tensor_rank, random_state=rands,
                        init=init, verbose=1, n_iter_max=40)
                except ValueError:
                    continue

                if rec_error[-1] < lowest_rec_error:
                    output_dict = {
                        'C_output': C_embed_wildcard_new,
                        'S1_output': S1_wildcard_new,
                        'S2_output': S2_wildcard_new,
                        'wildcard_output': None,
                    }
                    best_save_dict = deepcopy(output_dict)
                    lowest_rec_error = rec_error[-1]

            output_factor_dicts_w[output_tensor_rank] = best_save_dict

        timer.start()
        info_str = ''
        factor_dicts = dict()
        info_dicts = dict()

        rands = 0
        for rank in ranks:
            print('DECOMPOSING RANK: {}, TENSOR SIZE: {}'.format(rank, language_tensor.shape))

            lowest_rec_error = float('inf')
            best_save_dict = None
            for k in range(k_best):
                print('BEST of {}: {}'.format(k_best, k))
                rands = rands + k * 8
                try:
                    V_embed, S1, S2, rec_error = decompose_tensor_split_slot_3d_language(
                        language_tensor, language=language, word2idx=word2idx,
                        rank=rank, random_state=rands, n_iter_max=40,
                        init=init, verbose=1)
                except (ValueError, LinAlgError):
                    print('Value Error or LinAlgError')
                    continue

                if rec_error[-1] < lowest_rec_error:
                    save_dict = {
                        'V': V_embed,
                        'S1': S1,
                        'S2': S2,
                        'wildcard_mat': wildcard_mat,
                    }
                    best_save_dict = copy(save_dict)
                    lowest_rec_error = rec_error[-1]

            factor_dicts[rank] = best_save_dict
            info_dicts[rank] = lowest_rec_error.__round__(4)
            info_str += '{}-{}'.format(rank, info_dicts[rank])

        all_decomposed_automata[random_state] = [factor_dicts, output_factor_dicts, output_factor_dicts_w]

        timer.stop()

    pickle.dump(all_decomposed_automata,
                open(
                    '../../data/{dataset}/automata/IID.automata.{time_str}.{init}.{k}best.{n}states.{automata_name}.{split}splits.{info}.{rule_name}.pkl'. \
                    format(dataset=dataset_name,
                           time_str=time_str,
                           init=init,
                           k=k_best,
                           n=n_states,
                           automata_name=automata_name,
                           split=split_group,
                           info=info_str,
                           rule_name=rule_name), 'wb')
                )

    print('FINISHED')


def decompose_automata_single(rearranged_automata,
                              dataset_name,
                              automata_name,
                              split_group,
                              init='random',
                              k_best=2,
                              rule_name=''):

    timer.start()
    print('AUTOMATA TO TENSOR')
    print('Total States: {}'.format(len(rearranged_automata['states'])))

    data = load_slot_dataset(dataset_name, '../../data/')
    # first load vocabs
    word2idx = data['t2i']
    slot2idx = data['s2i']

    language_tensor, state2idx, wildcard_mat, output_mat, output_wildcard_vector, final_vector, start_vector, language = \
        dfa_to_tensor_slot_single(rearranged_automata, word2idx, slot2idx, dataset=dataset_name)

    _, _, _, output_mat_new, _, _, _, _ = \
        dfa_to_tensor_slot_single_wildcard(rearranged_automata, word2idx, slot2idx, dataset=dataset_name)

    timer.stop()

    print('DECOMPOSE SPLIT AUTOMATA')
    if dataset_name == 'MITR':
        ranks = [100, 150, 200]
    elif dataset_name == 'MITR-BIO':
        # ranks = [150, 200, 250]
        ranks = [250, 300]
    elif dataset_name == 'MITM-E-BIO':
        ranks = [200, 250, 300]
    elif dataset_name == 'ATIS-BIO':
        ranks = [100, 150, 200]
    elif dataset_name == 'ATIS-ZH-BIO':
        ranks = [300]
        # ranks = [250]
    elif dataset_name == 'SNIPS-BIO':
        ranks = [200, 250, 300]
    elif dataset_name == 'CONLL03-BIO':
        ranks = [100, 150, 200]
    else:
        ranks = [100, 150, 200]

    time_str = create_datetime_str()
    n_states = len(rearranged_automata['states'])

    output_factor_dicts = {
        'output_mat': output_mat,
        'output_wildcard_vector': output_wildcard_vector
    }

    output_factor_dicts_w = {
        'output_mat': output_mat_new,
        'output_wildcard_vector': output_wildcard_vector
    }

    all_decomposed_automata = dict()
    all_decomposed_automata['automata'] = rearranged_automata
    print(language)
    for random_state in range(4):

        timer.start()
        info_str = ''
        factor_dicts = dict()
        info_dicts = dict()

        rands = random_state
        for rank in ranks:
            print('DECOMPOSING RANK: {}, TENSOR SIZE: {}'.format(rank, language_tensor.shape))

            lowest_rec_error = float('inf')
            best_save_dict = None
            for k in range(k_best):
                print('BEST of {}: {}'.format(k_best, k))
                rands = rands + k * 8
                try:
                    V_embed, S1, S2, rec_error = decompose_tensor_split_slot_3d_language(
                        language_tensor, language=language, word2idx=word2idx,
                        rank=rank, random_state=rands, n_iter_max=32,
                        init=init, verbose=1)
                except (ValueError, LinAlgError):
                    print('Value Error or LinAlgError')
                    continue

                if rec_error[-1] < lowest_rec_error:
                    save_dict = {
                        'V': V_embed,
                        'S1': S1,
                        'S2': S2,
                        'wildcard_mat': wildcard_mat,
                    }
                    best_save_dict = copy(save_dict)
                    lowest_rec_error = rec_error[-1]

            factor_dicts[rank] = best_save_dict
            info_dicts[rank] = lowest_rec_error.__round__(4)
            info_str += '{}-{}'.format(rank, info_dicts[rank])

        all_decomposed_automata[random_state] = [factor_dicts, output_factor_dicts, output_factor_dicts_w]

        timer.stop()

    pickle.dump(all_decomposed_automata,
                open(
                    '../../data/{dataset}/automata/IIID.automata.{time_str}.{init}.{k}best.{n}states.{automata_name}.{split}splits.{info}.{rule_name}.pkl'. \
                    format(dataset=dataset_name,
                           time_str=time_str,
                           init=init,
                           k=k_best,
                           n=n_states,
                           automata_name=automata_name,
                           split=split_group,
                           info=info_str,
                           rule_name=rule_name), 'wb')
                )

    print('FINISHED')
