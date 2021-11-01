import pickle
from src_seq.utils import load_pkl
from copy import copy, deepcopy
from src_seq.wfa.fsa_to_tensor import drawGraphDict


def get_fr_states(automata, state):
    transitions = automata['transitions']
    fr_states = []
    for fr_state, to in sorted(transitions.items()):
        for to_state, to_edges in sorted(to.items()):
            if to_state == state:
                fr_states.append(fr_state)
    return fr_states


def get_to_states(automata, state):
    transitions = automata['transitions']
    to_states = []
    for fr_state, to in sorted(transitions.items()):
        for to_state, to_edges in sorted(to.items()):
            if to_state == state:
                to_states.append(to_state)
    return to_states


def get_all_fr_trans(automata, fr_states, to_state):
    transitions = automata['transitions']
    slot_dict = dict()
    for fr_state in fr_states:
        edges = transitions[fr_state][to_state]
        for edge in edges:
            word, slot = edge.split('<:>')
            if slot not in slot_dict:
                slot_dict[slot] = [(fr_state, to_state, word)]
            else:
                slot_dict[slot].append((fr_state, to_state, word))
    return slot_dict


def count_mul_inedge_node(automata):
    n_states = len(automata['states'])
    count = 0
    for state in automata['states']:
        fr_states = get_fr_states(automata, state)
        slot_dict = get_all_fr_trans(automata, fr_states, state)
        slot_set = set(slot_dict.keys())
        if len(slot_set) > 1:
            count += (len(slot_set) - 1)

    return count


def check_loop(slot_info):
    for slot in slot_info:
        if slot[0] == slot[1]:
            return True
    return False


def fix_inedge_node(automata):
    n_states = len(automata['states'])
    count = 0
    all_fix_node = dict()
    for state in automata['states']:
        fr_states = get_fr_states(automata, state)
        slot_dict = get_all_fr_trans(automata, fr_states, state)
        slot_set = set(slot_dict.keys())
        if len(slot_set) > 1:
            all_fix_node[state] = slot_set

    idx_state = n_states
    for fix_node, _ in all_fix_node.items():

        fr_states = get_fr_states(automata, fix_node)
        slot_dict = get_all_fr_trans(automata, fr_states, fix_node)

        is_final_state = fix_node in automata['finalstates']
        is_start_state = fix_node in automata['startstate']
        partitions = []
        all_slots = list(set(slot_dict.keys()))
        # partition the nodes
        slot_info = slot_dict[all_slots[0]]

        partitions.append((fix_node, all_slots[0], check_loop(slot_info)))

        for slot in all_slots[1:]:
            slot_info = slot_dict[slot]
            partitions.append((idx_state, slot, check_loop(slot_info)))
            idx_state += 1

        print(partitions)

        if fix_node in automata['transitions']:
            old_transitions = deepcopy(automata['transitions'][fix_node])
            for new_state, slot, is_loop in partitions:

                # remove the self loop transitions that is invalid
                new_edges = dict()
                # deal with self loop node, when we encounter self loop node that is not oo, do not add that

                for to_state, edges in old_transitions.items():
                    # cancel all the loops
                    if to_state == fix_node:
                        if is_loop:
                            new_edges[new_state] = set([i for i in edges if i.split('<:>')[1] == slot])
                        else:
                            pass

                    else:
                        new_edges[to_state] = deepcopy(edges)
                automata['transitions'][new_state] = new_edges

        for new_state, slot, is_loop in partitions:

            if new_state not in automata['states']: # already in

                automata['states'].add(new_state)
                if is_final_state:
                    automata['finalstates'].append(new_state)
                if is_start_state:
                    automata['startstate'].append(new_state)

                # add the transitions for the given slot
                for fr, _, inp in slot_dict[slot]:
                    if new_state not in automata['transitions'][fr]:
                        automata['transitions'][fr][new_state] = set(['{}<:>{}'.format(inp, slot)])
                    else:
                        automata['transitions'][fr][new_state].add('{}<:>{}'.format(inp, slot))

            else:
                # remove other slots edges for the existing node (except wildcard)
                for s in all_slots:
                    if s != slot:
                        for fr, to, inp in slot_dict[s]:
                            if new_state in automata['transitions'][fr]:
                                automata['transitions'][fr][new_state].discard('{}<:>{}'.format(inp, s))

        # add selfloop transitions, non-selfloop node to all selfloop node
        for new_state, slot, is_loop in partitions:
            if is_loop:
                for state, slot, is_loop in partitions:
                    if state != new_state:
                        automata['transitions'][state][new_state] = automata['transitions'][new_state][new_state]


    return automata




if __name__ == '__main__':
    autoamta_name = '../../data/MITM-E-BIO/automata/automata.random.5splits.dict'
    automata = load_pkl(autoamta_name)
    drawGraphDict(automata, '../../data/MITM-E-BIO/automata/test.1')
    print(count_mul_inedge_node(automata))

    new_automata = fix_inedge_node(automata)
    pickle.dump(new_automata, open(autoamta_name+'.independent.III', 'wb'))
    drawGraphDict(new_automata, '../../data/MITM-E-BIO/automata/test.2')
    print(count_mul_inedge_node(new_automata))

    # autoamta_name = '../../data/ATIS-BIO/automata/automata.ATIS-random.5splits.dict'
    # drawGraphDict(load_pkl(autoamta_name), '../../data/ATIS-BIO/automata/test.1')
    #
    # new_automata = fix_inedge_node(load_pkl(autoamta_name))
    # pickle.dump(new_automata, open(autoamta_name+'.independent.III', 'wb'))
    # drawGraphDict(new_automata, '../../data/ATIS-BIO/automata/test.2')

    # autoamta_name = '../../data/MITR-BIO/automata/automata.MITR.5splits.dict'
    # drawGraphDict(load_pkl(autoamta_name), '../../data/MITR-BIO/automata/test.1')
    #
    # new_automata = fix_inedge_node(load_pkl(autoamta_name))
    # pickle.dump(new_automata, open(autoamta_name+'.independent.III', 'wb'))
    # drawGraphDict(new_automata, '../../data/MITR-BIO/automata/test.2')
    # print(count_mul_inedge_node(new_automata))
    # print(new_automata)
    pass
