import pickle
import argparse
from data import load_pkl, load_classification_dataset, decompose_tensor_split
import numpy as np
import random
from copy import deepcopy
from rules.fsa_to_tensor import dfa_to_tensor
from src_seq.utils import mkdir


def copy_subtype(automata, subtype, copyidx):
    new_automata = deepcopy(automata)
    # copy all out edges
    new_automata['transitions'][subtype] = automata['transitions'][copyidx]
    # copy all out edges for others
    for from_state, to_states in automata['transitions'].items():
        for to_state, edges in to_states.items():
            if to_state == copyidx:
                new_automata['transitions'][from_state][subtype] = edges

    return new_automata

def generalize_tensor_with_subtype(args):
    np.random.seed(0)
    random.seed(0)
    # load original automata
    automata_dicts = load_pkl(args.automata_path)
    automata = automata_dicts['automata']

    # option 1: only update internal node
    # find all internal state (exclude start and final)
    mode = 'interm_small'
    all_copyable_states = set(automata['states']) - set([automata['startstate']]) - set(automata['finalstates'])
    num_copy_states = int( len(all_copyable_states)*args.addtional_subtype_state_portion )
    random_pick_copy_state = np.random.choice(list(all_copyable_states), size=num_copy_states, replace=False)

    # create subtype indices
    subtype_idxs = set([i + max(automata['states']) for i in range(1, num_copy_states+1)])

    # add the random picked subtypes into the automata
    automata['subtypes'] = subtype_idxs
    automata['states'].update(subtype_idxs)

    subtype_idxs = list(subtype_idxs)
    # start add transitions
    assert len(subtype_idxs) == len(random_pick_copy_state)
    for i in range(len(subtype_idxs)):
        subtype = subtype_idxs[i]
        copyidx = random_pick_copy_state[i]

        # copy all out-edges and in edges
        automata = copy_subtype(automata, subtype, copyidx)

    dataset = load_classification_dataset(args.dataset)
    word2idx = dataset['t2i']
    language_tensor, state2idx, wildcard_mat, language = dfa_to_tensor(automata, word2idx, subtype=True)

    print('DECOMPOSE SPLIT AUTOMATA')
    mkdir('../data/{}/automata'.format(args.dataset))
    ranks = [150, 200]
    non_negative = [False]
    for rank in ranks:
        for nonn in non_negative:
            print('DECOMPOSING RANK: {}, NON-NEGATIV: {}, TENSOR SIZE: {}'.format(rank, nonn, language_tensor.shape))
            V_embed_split, D1_split, D2_split, rec_error = decompose_tensor_split(language_tensor, language, word2idx, rank)
            save_dict = {
                'automata': automata,
                'V':V_embed_split,
                'D1': D1_split,
                'D2': D2_split,
                'language': language,
                'wildcard_mat': wildcard_mat,
            }
            pickle.dump(save_dict, open('../data/{}/automata/automata.split.{}.{}.{:.4f}.{}.{}.pkl'.format(args.dataset, rank, nonn, rec_error[-1], args.addtional_subtype_state_portion, mode), 'wb'))

    print('FINISHED')

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--automata_path', type=str, default='../jupyter/rule/ATIS/AUTOMATA/150.split.pkl', help='automata path')
    # parser.add_argument('--dataset', type=str, default='ATIS', help='automata path')

    parser.add_argument('--automata_path', type=str, default='../data/TREC/automata/automata.split.200.False.0.0016.pkl', help='automata path')
    parser.add_argument('--dataset', type=str, default='TREC', help='automata path')

    parser.add_argument('--addtional_subtype_state_portion', type=float, default=0.4, help='additional subtype internal states')
    args = parser.parse_args()
    generalize_tensor_with_subtype(args)
