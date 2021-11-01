from src_seq.utils import load_pkl
from copy import copy, deepcopy
import pickle

def check_independent(automata):
    """
    :param automata:
    :return:
    """
    """
    :param automata_name:
    :return:
    """

    transitions = automata['transitions']
    print('total states: {}'.format(len(automata['states'])))

    all_dependent_pairs = dict()

    for fr_state, to in transitions.items():
        for to_state, to_edges in to.items():

            tag_word_dict = dict()

            for edge in to_edges:  # set
                word, tag = edge.split('<:>')
                word = word.lower()
                tag = tag.lower()

                if tag == 'oo':
                    continue

                if tag not in tag_word_dict:
                    tag_word_dict[tag] = [word]
                else:
                    tag_word_dict[tag].append(word)

            if len(tag_word_dict.keys()) > 1:
                all_dependent_pairs[(fr_state, to_state)] = \
                    deepcopy(tag_word_dict)

    print('ALL DEPENDENT PAIRS: {}'.format(len(all_dependent_pairs)))

    return all_dependent_pairs


def check_independent_from_file(automata_name):
    """
    :param automata_name:
    :return:
    """

    automata = load_pkl(automata_name)

    transitions = automata['transitions']
    print('total states: {}'.format(len(automata['states'])))

    all_dependent_pairs = dict()

    for fr_state, to in transitions.items():
        for to_state, to_edges in to.items():

            tag_word_dict = dict()

            for edge in to_edges: # set
                word, tag = edge.split('<:>')
                word = word.lower()
                tag = tag.lower()

                if tag == 'oo':
                    continue

                if tag not in tag_word_dict:
                    tag_word_dict[tag] = [word]
                else:
                    tag_word_dict[tag].append(word)

            if len(tag_word_dict.keys()) > 1:
                all_dependent_pairs[(fr_state, to_state)] = \
                    deepcopy(tag_word_dict)

    print('ALL DEPENDENT PAIRS: {}'.format(len(all_dependent_pairs)))

    return all_dependent_pairs


def fix_some_dependent(automata_name):

    all_dependent_pairs = check_independent_from_file(automata_name)
    automata = load_pkl(automata_name)

    n_states = len(automata['states'])

    fr_set = set([i[0] for i in all_dependent_pairs.keys()])

    for (fr_state, to_state), tag_word_dict in all_dependent_pairs.items():
        if to_state in automata['finalstates']:

            # One level fixing, should be recursive.
            cp = None
            if to_state in automata['transitions']:
                if to_state not in fr_set:
                    cp = deepcopy(automata['transitions'][to_state])

            for tag in list(tag_word_dict.keys())[1:]:
                words = tag_word_dict[tag]
                # create a new final states
                automata['states'].add(n_states)
                automata['finalstates'].append(n_states)
                automata['transitions'][fr_state][n_states] = set()
                for word in words:
                    recovered_word_tag = '{}<:>{}'.format(word, tag)
                    # remove old transitions
                    automata['transitions'][fr_state][to_state].remove(recovered_word_tag)
                    # add new transitions
                    automata['transitions'][fr_state][n_states].add(recovered_word_tag)

                if cp is not None:
                    automata['transitions'][n_states] = cp

                n_states += 1

    return automata


def fix_all_dependent(automata):

    n_states = len(automata['states'])
    all_dependent_pairs = check_independent(automata)

    if len(all_dependent_pairs) == 0:
        return automata

    for (fr_state, to_state), tag_word_dict in all_dependent_pairs.items():
        # One level fixing, should be recursive.
        cp = None
        if to_state in automata['transitions']:
            cp = deepcopy(automata['transitions'][to_state])

        for tag in list(tag_word_dict.keys())[1:]:
            words = tag_word_dict[tag]
            # create a new final states
            automata['states'].add(n_states)
            if to_state in automata['finalstates']:
                automata['finalstates'].append(n_states)
            automata['transitions'][fr_state][n_states] = set()
            for word in words:
                recovered_word_tag = '{}<:>{}'.format(word, tag)
                # remove old transitions
                automata['transitions'][fr_state][to_state].remove(recovered_word_tag)
                # add new transitions
                automata['transitions'][fr_state][n_states].add(recovered_word_tag)

            if cp is not None:
                automata['transitions'][n_states] = cp

            n_states += 1

        if cp is not None:
            break

    return fix_all_dependent(deepcopy(automata))

if __name__ == '__main__':
    autoamta_name = '../../data/MITM-E-BIO/automata/automata.MITM-E-random.1000splits.dict'
    new_automata = fix_all_dependent(load_pkl(autoamta_name))

    pickle.dump(new_automata, open(autoamta_name+'.independent', 'wb'))