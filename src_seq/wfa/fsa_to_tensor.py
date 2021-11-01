import numpy as np
from os import popen
from typing import Dict

punctuations = {
    ',', '，', ':', '：', '!', '！', '《', '》', '。', '；', '.', '(', ')', '（', '）',
    '|', '?', '"'
}


def create_toy_tensor(idx2word, word2idx):

    tensor = np.zeros((len(idx2word), 6, 6))
    tensor[word2idx['flights'], 0, 1] = 1
    tensor[word2idx['from'], 1, 2] = 1
    tensor[word2idx['california'], 2, 3] = 1
    tensor[word2idx['california'], 4, 5] = 1
    tensor[word2idx['carolina'], 2, 3] = 1
    tensor[word2idx['carolina'], 4, 5] = 1
    tensor[word2idx['to'], 3, 4] = 1
    tensor[:, 5, 5] = 1
    tensor[:, 5, 5] = 1
    tensor[:, 0, 0] = 1
    tensor[:, 0, 0] = 1
    tensor[word2idx['flights'], 0, 0] = 0

    return tensor


def create_toy_tensor_multiple_final(idx2word, word2idx):

    tensor = np.zeros((len(idx2word), 3, 3))
    tensor[word2idx['flights'], 0, 1] = 1
    tensor[word2idx['flights'], 1, 1] = 1
    tensor[word2idx['from'], 0, 2] = 1

    # tensor[:, 5, 5] = 1
    # tensor[:, 5, 5] = 1
    tensor[:, 0, 0] = 1
    tensor[:, 0, 0] = 1
    tensor[word2idx['flights'], 0, 0] = 0
    tensor[word2idx['from'], 0, 0] = 0

    return tensor


def is_small_pos_number(token):
    token = token.replace('.', '', 1)
    try:
        token_num = int(token)
        if (token_num < 25) and (token_num >= 0):
            return True
        else:
            return False
    except:
        return False


def is_number(token):
    return token.replace('.', '', 1).isdigit()


def get_num_punct(word2idx, dataset):
    if dataset == 'MITR-BIO':
        number_idxs = {word: idx for word, idx in word2idx.items() if is_small_pos_number(word)}
        punct_idxs = {word: idx for word, idx in word2idx.items() if is_punct(word)}
    elif dataset == 'MITM-E-BIO':
        number_idxs = {word: idx for word, idx in word2idx.items() if is_number(word)}
        punct_idxs = {word: idx for word, idx in word2idx.items() if is_punct(word)}
    elif dataset == 'ATIS-BIO':
        number_idxs = {word: idx for word, idx in word2idx.items() if is_number(word)}
        punct_idxs = {word: idx for word, idx in word2idx.items() if is_punct(word)}
    elif dataset == 'ATIS-ZH-BIO':
        number_idxs = {word: idx for word, idx in word2idx.items() if is_number(word)}
        punct_idxs = {word: idx for word, idx in word2idx.items() if is_punct(word)}
    elif dataset == 'SNIPS-BIO':
        number_idxs = {word: idx for word, idx in word2idx.items() if is_number(word)}
        punct_idxs = {word: idx for word, idx in word2idx.items() if is_punct(word)}
    else:
        raise NotImplementedError()

    return number_idxs, punct_idxs


def is_punct(token):
    return token in punctuations


def get_val(slot):
    return 1

def dfa_to_tensor_slot(automata, word2idx: Dict[str, int], slot2idx: Dict[str, int], weigh_o=1):
    """
    Parameters
    ----------
    automata: Automata.to_dict()
    word2idx

    Returns
    -------
    tensor: tensor for language
    state2idx: state to idx
    wildcard_mat: matrix for wildcard
    language: set for language
    """

    # TODO: need implementation
    all_states = list(automata['states'])
    state2idx = {
        state: idx for idx, state in enumerate(all_states)
    }

    number_idxs = {word: idx for word, idx in word2idx.items() if is_small_pos_number(word)}
    punct_idxs = {word: idx for word, idx in word2idx.items() if is_punct(word)}

    max_states = len(automata['states'])
    language_tensor = np.zeros((len(word2idx), len(slot2idx), max_states, max_states))
    wildcard_tensor = np.zeros((len(slot2idx), max_states, max_states))

    language = set([])

    for fr_state, to in sorted(automata['transitions'].items()):
        for to_state, to_edges in sorted(to.items()):
            for edge in to_edges:

                word, slot = edge.split('<:>')
                if slot == 'oo':
                    slot_idxs = list(slot2idx.values())
                else:
                    slot_idxs = [slot2idx[slot]]

                val = get_val(slot)


                if word == '&': # punctuations
                    language_tensor[list(punct_idxs.values()), slot_idxs, state2idx[fr_state], state2idx[to_state]] = val
                    language.update(punct_idxs.keys())

                elif word == '%': # digits
                    language_tensor[list(number_idxs.values()), slot_idxs, state2idx[fr_state], state2idx[to_state]] = val
                    language.update(number_idxs.keys())

                elif word == "$":

                    if weigh_o == 1:
                        # Mode 2
                        wildcard_tensor[slot_idxs, state2idx[fr_state], state2idx[to_state]] = val / len(slot_idxs)
                        wildcard_tensor[slot2idx['o'], state2idx[fr_state], state2idx[to_state]] = 1

                        # Mode 3 After Mode 2, normalize it
                        wildcard_tensor[:, state2idx[fr_state], state2idx[to_state]] /= np.sum(wildcard_tensor[:, state2idx[fr_state], state2idx[to_state]])

                    elif weigh_o == 0:
                        # Mode 1
                        wildcard_tensor[slot_idxs, state2idx[fr_state], state2idx[to_state]] = val

                    elif weigh_o == 2:
                        wildcard_tensor[slot_idxs, state2idx[fr_state], state2idx[to_state]] = val / len(slot_idxs)


                else:
                    if word in word2idx:
                        language_tensor[word2idx[word], slot_idxs, state2idx[fr_state], state2idx[to_state]] = val
                        language.add(word)
                    else:
                        print('OOV word: {} in rule'.format(word))
                        pass


    final_vector = np.zeros(max_states)
    final_vector[automata['finalstates']] = 1

    return language_tensor, state2idx, wildcard_tensor, final_vector, list(language)


def dfa_to_tensor_slot_new(
        automata,
        word2idx: Dict[str, int],
        slot2idx: Dict[str, int],
        dataset='MITR-BIO'):
    """
    Parameters
    ----------
    automata: Automata.to_dict()
    word2idx: word to index
    slot2idx: slot to index
    dataset: dataset name

    Returns
    -------
    tensor: tensor for language
    state2idx: state to idx
    wildcard_mat: matrix for wildcard
    language: set for language
    """

    # TODO: need implementation
    all_states = list(automata['states'])
    state2idx = {
        state: idx for idx, state in enumerate(all_states)
    }

    number_idxs, punct_idxs = get_num_punct(word2idx, dataset)

    max_states = len(automata['states'])
    language_tensor = np.zeros((len(word2idx), len(slot2idx), max_states, max_states))
    wildcard_tensor = np.zeros((len(slot2idx), max_states, max_states))
    wildcard_wildcard_tensor = np.zeros((max_states, max_states))

    language = set([])

    for fr_state, to in sorted(automata['transitions'].items()):
        for to_state, to_edges in sorted(to.items()):
            for edge in to_edges:

                word, slot = edge.split('<:>')

                if slot == 'oo':
                    slot_idxs = list(slot2idx.values())
                    assert word == '$'
                else:
                    slot_idxs = [slot2idx[slot]]

                val = get_val(slot)


                if word == '&': # punctuations
                    language_tensor[list(punct_idxs.values()), slot_idxs, state2idx[fr_state], state2idx[to_state]] = val
                    language.update(punct_idxs.keys())

                elif word == '%': # digits
                    language_tensor[list(number_idxs.values()), slot_idxs, state2idx[fr_state], state2idx[to_state]] = val
                    language.update(number_idxs.keys())

                elif word == "$":
                    if slot != 'oo':
                        wildcard_tensor[slot_idxs, state2idx[fr_state], state2idx[to_state]] = val
                    else: # handle the $<:>OO case
                        wildcard_wildcard_tensor[state2idx[fr_state], state2idx[to_state]] = val
                else:
                    if word in word2idx:
                        language_tensor[word2idx[word], slot_idxs, state2idx[fr_state], state2idx[to_state]] = val
                        language.add(word)
                    else:
                        print('OOV word: {} in rule'.format(word))
                        pass

    final_vector = np.zeros(max_states)
    final_vector[automata['finalstates']] = 1

    start_vector = np.zeros(max_states)
    start_vector[automata['startstate']] = 1

    print("LANGUAGE SET SIZE: {}".format(len(language)))

    return language_tensor, state2idx, wildcard_tensor, wildcard_wildcard_tensor, \
           final_vector, start_vector, list(language)


def dfa_to_tensor_slot_independent(
        automata,
        word2idx: Dict[str, int],
        slot2idx: Dict[str, int],
        dataset='MITR-BIO'):
    """
    Parameters
    """

    # TODO: need implementation
    all_states = list(automata['states'])
    state2idx = {
        state: idx for idx, state in enumerate(all_states)
    }

    number_idxs, punct_idxs = get_num_punct(word2idx, dataset)

    max_states = len(automata['states'])
    language_tensor = np.zeros((len(word2idx), max_states, max_states)) # V x S x S
    language_wildcard_mat = np.zeros((max_states, max_states)) # S x S
    output_tensor = np.zeros((len(slot2idx), max_states, max_states)) # C x S x S
    output_wildcard_mat = np.zeros((max_states, max_states))

    language = set([])

    for fr_state, to in sorted(automata['transitions'].items()):
        for to_state, to_edges in sorted(to.items()):
            for edge in to_edges:

                word, slot = edge.split('<:>')

                val = get_val(slot)


                if slot != 'oo':
                    slot_idxs = [slot2idx[slot]]
                    output_tensor[slot_idxs, state2idx[fr_state], state2idx[to_state]] = val
                else:
                    output_wildcard_mat[state2idx[fr_state], state2idx[to_state]] = val

                if word == '&': # punctuations
                    language_tensor[list(punct_idxs.values()), state2idx[fr_state], state2idx[to_state]] = val
                    language.update(punct_idxs.keys())

                elif word == '%': # digits
                    language_tensor[list(number_idxs.values()), state2idx[fr_state], state2idx[to_state]] = val
                    language.update(number_idxs.keys())

                elif word == "$":
                    language_wildcard_mat[state2idx[fr_state], state2idx[to_state]] = val
                else:
                    if word in word2idx:
                        language_tensor[word2idx[word], state2idx[fr_state], state2idx[to_state]] = val
                        language.add(word)
                    else:
                        print('OOV word: {} in rule'.format(word))
                        pass

    final_vector = np.zeros(max_states)
    final_vector[automata['finalstates']] = 1

    start_vector = np.zeros(max_states)
    start_vector[automata['startstate']] = 1

    print("LANGUAGE SET SIZE: {}".format(len(language)))

    return language_tensor, state2idx, language_wildcard_mat, output_tensor, output_wildcard_mat, final_vector, \
           start_vector, list(language)


def dfa_to_tensor_slot_single(
        automata,
        word2idx: Dict[str, int],
        slot2idx: Dict[str, int],
        dataset='MITR-BIO'):
    """
    Parameters
    """

    # TODO: need implementation
    all_states = list(automata['states'])
    state2idx = {
        state: idx for idx, state in enumerate(all_states)
    }

    number_idxs, punct_idxs = get_num_punct(word2idx, dataset)

    max_states = len(automata['states'])
    language_tensor = np.zeros((len(word2idx), max_states, max_states)) # V x S x S
    language_wildcard_mat = np.zeros((max_states, max_states)) # S x S

    output_wildcard_vector = np.zeros((max_states))
    output_mat = np.zeros((len(slot2idx), max_states))

    language = set([])

    for fr_state, to in sorted(automata['transitions'].items()):
        for to_state, to_edges in sorted(to.items()):
            for edge in to_edges:

                word, slot = edge.split('<:>')

                val = get_val(slot)

                if slot != 'oo':
                    slot_idxs = [slot2idx[slot]]
                    output_mat[slot_idxs, state2idx[to_state]] = val
                else:
                    output_wildcard_vector[state2idx[to_state]] = val

                if word == '&': # punctuations
                    language_tensor[list(punct_idxs.values()), state2idx[fr_state], state2idx[to_state]] = val
                    language.update(punct_idxs.keys())
                elif word == '%': # digits
                    language_tensor[list(number_idxs.values()), state2idx[fr_state], state2idx[to_state]] = val
                    language.update(number_idxs.keys())
                elif word == "$":
                    language_wildcard_mat[state2idx[fr_state], state2idx[to_state]] = val
                else:
                    if word in word2idx:
                        language_tensor[word2idx[word], state2idx[fr_state], state2idx[to_state]] = val
                        language.add(word)
                    else:
                        print('OOV word: {} in rule'.format(word))
                        pass

    final_vector = np.zeros(max_states)
    final_vector[automata['finalstates']] = 1

    start_vector = np.zeros(max_states)
    start_vector[automata['startstate']] = 1

    print("LANGUAGE SET SIZE: {}".format(len(language)))

    return language_tensor, state2idx, language_wildcard_mat, output_mat, output_wildcard_vector, final_vector, \
           start_vector, list(language)


def dfa_to_tensor_slot_new_wildcard(
        automata,
        word2idx: Dict[str, int],
        slot2idx: Dict[str, int],
        dataset='MITR-BIO'):
    """
    Parameters
    ----------
    automata: Automata.to_dict()
    word2idx: word to index
    slot2idx: slot to index
    dataset: dataset name

    Returns
    -------
    tensor: tensor for language
    state2idx: state to idx
    wildcard_mat: matrix for wildcard
    language: set for language
    """

    # TODO: need implementation
    all_states = list(automata['states'])
    state2idx = {
        state: idx for idx, state in enumerate(all_states)
    }

    number_idxs, punct_idxs = get_num_punct(word2idx, dataset)

    max_states = len(automata['states'])
    language_tensor = np.zeros((len(word2idx), len(slot2idx)+1, max_states, max_states))
    wildcard_tensor = np.zeros((len(slot2idx)+1, max_states, max_states))
    wildcard_wildcard_tensor = np.zeros((max_states, max_states)) # None

    language = set([])

    for fr_state, to in sorted(automata['transitions'].items()):
        for to_state, to_edges in sorted(to.items()):
            for edge in to_edges:

                word, slot = edge.split('<:>')

                if slot == 'oo':
                    slot_idxs = [len(slot2idx)]
                    assert word == '$'
                else:
                    slot_idxs = [slot2idx[slot]]

                val = get_val(slot)

                if word == '&': # punctuations
                    language_tensor[list(punct_idxs.values()), slot_idxs, state2idx[fr_state], state2idx[to_state]] = val
                    language.update(punct_idxs.keys())

                elif word == '%': # digits
                    language_tensor[list(number_idxs.values()), slot_idxs, state2idx[fr_state], state2idx[to_state]] = val
                    language.update(number_idxs.keys())

                elif word == "$":
                    wildcard_tensor[slot_idxs, state2idx[fr_state], state2idx[to_state]] = val

                else:
                    if word in word2idx:
                        language_tensor[word2idx[word], slot_idxs, state2idx[fr_state], state2idx[to_state]] = val
                        language.add(word)
                    else:
                        print('OOV word: {} in rule'.format(word))
                        pass

    final_vector = np.zeros(max_states)
    final_vector[automata['finalstates']] = 1

    start_vector = np.zeros(max_states)
    start_vector[automata['startstate']] = 1

    print("LANGUAGE SET SIZE: {}".format(len(language)))

    return language_tensor, state2idx, wildcard_tensor, wildcard_wildcard_tensor, \
           final_vector, start_vector, list(language)


def dfa_to_tensor_slot_independent_wildcard(
        automata,
        word2idx: Dict[str, int],
        slot2idx: Dict[str, int],
        dataset='MITR-BIO'):
    """
    Parameters
    """

    # TODO: need implementation
    all_states = list(automata['states'])
    state2idx = {
        state: idx for idx, state in enumerate(all_states)
    }

    number_idxs, punct_idxs = get_num_punct(word2idx, dataset)

    max_states = len(automata['states'])
    language_tensor = np.zeros((len(word2idx), max_states, max_states)) # V x S x S
    language_wildcard_mat = np.zeros((max_states, max_states)) # S x S
    output_tensor = np.zeros((len(slot2idx)+1, max_states, max_states)) # C x S x S
    output_wildcard_mat = None

    language = set([])

    for fr_state, to in sorted(automata['transitions'].items()):
        for to_state, to_edges in sorted(to.items()):
            for edge in to_edges:

                word, slot = edge.split('<:>')

                val = get_val(slot)

                if slot != 'oo':
                    slot_idxs = [slot2idx[slot]]
                else:
                    slot_idxs = [len(slot2idx)]

                output_tensor[slot_idxs, state2idx[fr_state], state2idx[to_state]] = val

                if word == '&': # punctuations
                    language_tensor[list(punct_idxs.values()), state2idx[fr_state], state2idx[to_state]] = val
                    language.update(punct_idxs.keys())

                elif word == '%': # digits
                    language_tensor[list(number_idxs.values()), state2idx[fr_state], state2idx[to_state]] = val
                    language.update(number_idxs.keys())

                elif word == "$":
                    language_wildcard_mat[state2idx[fr_state], state2idx[to_state]] = val
                else:
                    if word in word2idx:
                        language_tensor[word2idx[word], state2idx[fr_state], state2idx[to_state]] = val
                        language.add(word)
                    else:
                        print('OOV word: {} in rule'.format(word))
                        pass

    final_vector = np.zeros(max_states)
    final_vector[automata['finalstates']] = 1

    start_vector = np.zeros(max_states)
    start_vector[automata['startstate']] = 1

    print("LANGUAGE SET SIZE: {}".format(len(language)))

    return language_tensor, state2idx, language_wildcard_mat, output_tensor, output_wildcard_mat, final_vector, \
           start_vector, list(language)


def dfa_to_tensor_slot_single_wildcard(
        automata,
        word2idx: Dict[str, int],
        slot2idx: Dict[str, int],
        dataset='MITR-BIO'):
    """
    Parameters
    """

    all_states = list(automata['states'])
    state2idx = {
        state: idx for idx, state in enumerate(all_states)
    }

    number_idxs, punct_idxs = get_num_punct(word2idx, dataset)

    max_states = len(automata['states'])
    language_tensor = np.zeros((len(word2idx), max_states, max_states)) # V x S x S
    language_wildcard_mat = np.zeros((max_states, max_states)) # S x S

    output_wildcard_vector = np.zeros((max_states))
    output_mat = np.zeros((len(slot2idx)+1, max_states))

    language = set([])

    for fr_state, to in sorted(automata['transitions'].items()):
        for to_state, to_edges in sorted(to.items()):
            for edge in to_edges:

                word, slot = edge.split('<:>')

                val = get_val(slot)

                if slot != 'oo':
                    slot_idxs = [slot2idx[slot]]
                else:
                    slot_idxs = [len(slot2idx)]
                output_mat[slot_idxs, state2idx[to_state]] = val

                if word == '&': # punctuations
                    language_tensor[list(punct_idxs.values()), state2idx[fr_state], state2idx[to_state]] = val
                    language.update(punct_idxs.keys())

                elif word == '%': # digits
                    language_tensor[list(number_idxs.values()), state2idx[fr_state], state2idx[to_state]] = val
                    language.update(number_idxs.keys())

                elif word == "$":
                    language_wildcard_mat[state2idx[fr_state], state2idx[to_state]] = val
                else:
                    if word in word2idx:
                        language_tensor[word2idx[word], state2idx[fr_state], state2idx[to_state]] = val
                        language.add(word)
                    else:
                        print('OOV word: {} in rule'.format(word))
                        pass

    final_vector = np.zeros(max_states)
    final_vector[automata['finalstates']] = 1

    start_vector = np.zeros(max_states)
    start_vector[automata['startstate']] = 1

    print("LANGUAGE SET SIZE: {}".format(len(language)))

    return language_tensor, state2idx, language_wildcard_mat, output_mat, output_wildcard_vector, final_vector, \
           start_vector, list(language)


class Automata:
    """class to represent an Automata"""

    def __init__(self, language=set(['0', '1'])):
        self.states = set()
        self.startstate = None
        self.finalstates = []
        self.finalstates_label = {}
        self.transitions = dict()
        self.language = language

    def to_dict(self):
        return {
            'states': self.states,
            'startstate': self.startstate,
            'finalstates': self.finalstates,
            'transitions': self.transitions,
            'language': self.language,
            'finalstates_label': self.finalstates_label
        }

    def setstartstate(self, state):
        self.startstate = state
        self.states.add(state)

    def addfinalstates(self, state):
        if isinstance(state, int):
            state = [state]
        for s in state:
            if s not in self.finalstates:
                self.finalstates.append(s)

    def addfinalstates_label(self, state, label):
        if isinstance(state, int):
            state = [state]
        assert label not in self.finalstates_label
        self.finalstates_label[label] = state

    def addtransition(self, fromstate, tostate, inp):
        if isinstance(inp, str):
            inp = set([inp])
        self.states.add(fromstate)
        self.states.add(tostate)
        if fromstate in self.transitions:
            if tostate in self.transitions[fromstate]:
                self.transitions[fromstate][tostate] = self.transitions[fromstate][tostate].union(inp)
            else:
                self.transitions[fromstate][tostate] = inp
        else:
            self.transitions[fromstate] = {tostate: inp}

    def addtransition_dict(self, transitions):
        for fromstate, tostates in transitions.items():
            for state in tostates:
                self.addtransition(fromstate, state, tostates[state])

    def getDotFile(self):
        dotFile = "digraph DFA {\nrankdir=LR\n"
        if len(self.states) != 0:
            dotFile += "root=s1\nstart [shape=point]\nstart->s%d\n" % self.startstate
            for state in self.states:
                if state in self.finalstates:
                    dotFile += "s%d [shape=doublecircle]\n" % state
                else:
                    dotFile += "s%d [shape=circle]\n" % state
            for fromstate, tostates in self.transitions.items():
                for state in tostates:
                    for char in tostates[state]:
                        dotFile += 's%d->s%d [label="%s"]\n' % (fromstate, state, char)
        dotFile += "}"
        return dotFile



class AutomataMultipleStart:
    """class to represent an Automata"""

    def __init__(self, language=set(['0', '1'])):
        self.states = set()
        self.startstate = []
        self.finalstates = []
        self.finalstates_label = {}
        self.transitions = dict()
        self.language = language

    def to_dict(self):
        return {
            'states': self.states,
            'startstate': self.startstate,
            'finalstates': self.finalstates,
            'transitions': self.transitions,
            'language': self.language,
            'finalstates_label': self.finalstates_label
        }

    def setstartstate(self, state):
        if isinstance(state, int):
            state = [state]
        for s in state:
            if s not in self.finalstates:
                self.startstate.append(s)
                self.states.add(s)

    def addfinalstates(self, state):
        if isinstance(state, int):
            state = [state]
        for s in state:
            if s not in self.finalstates:
                self.finalstates.append(s)

    def addfinalstates_label(self, state, label):
        if isinstance(state, int):
            state = [state]
        assert label not in self.finalstates_label
        self.finalstates_label[label] = state

    def addtransition(self, fromstate, tostate, inp):
        if isinstance(inp, str):
            inp = set([inp])
        self.states.add(fromstate)
        self.states.add(tostate)
        if fromstate in self.transitions:
            if tostate in self.transitions[fromstate]:
                self.transitions[fromstate][tostate] = self.transitions[fromstate][tostate].union(inp)
            else:
                self.transitions[fromstate][tostate] = inp
        else:
            self.transitions[fromstate] = {tostate: inp}

    def addtransition_dict(self, transitions):
        for fromstate, tostates in transitions.items():
            for state in tostates:
                self.addtransition(fromstate, state, tostates[state])

    def getDotFile(self):
        dotFile = "digraph DFA {\nrankdir=LR\n"
        if len(self.states) != 0:

            for state in self.states:
                if state in self.finalstates:
                    dotFile += "s%d [shape=doublecircle]\n" % state
                elif state in self.startstate:
                    dotFile += "root=s1\nstart [shape=point]\nstart->s%d\n" % state
                else:
                    dotFile += "s%d [shape=circle]\n" % state
            for fromstate, tostates in self.transitions.items():
                for state in tostates:
                    for char in tostates[state]:
                        dotFile += 's%d->s%d [label="%s"]\n' % (fromstate, state, char)
        dotFile += "}"
        return dotFile

def drawGraph(automata, file="",):
    """From https://github.com/max99x/automata-editor/blob/master/util.py"""
    f = popen(r"dot -Tpng -o %s.png" % file, 'w')
    try:
        f.write(automata.getDotFile())
    except:
        raise BaseException("Error creating graph")
    finally:
        f.close()


def drawGraphDict(automata, file="",):
    dotFile = "digraph DFA {\nrankdir=LR\n"
    if len(automata['states']) != 0:
        for state in automata['startstate']:
            dotFile += "root=s1\nstart [shape=point]\nstart->s%d\n" % state
        for state in automata['states']:
            if state in automata['finalstates']:
                dotFile += "s%d [shape=doublecircle]\n" % state
            else:
                dotFile += "s%d [shape=circle]\n" % state
        for fromstate, tostates in automata['transitions'].items():
            for state in tostates:
                for char in tostates[state]:
                    dotFile += 's%d->s%d [label="%s"]\n' % (fromstate, state, char)
    dotFile += "}"

    f = popen(r"dot -Tpng -o %s.png" % file, 'w')
    try:
        f.write(dotFile)
    except:
        raise BaseException("Error creating graph")
    finally:
        f.close()
