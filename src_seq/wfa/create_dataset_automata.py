import sys
sys.path.append('../../')
sys.path.append('../')
from src_seq.load_data_and_rules import *
from src_seq.wfa.fsa_to_tensor import drawGraph, AutomataMultipleStart, drawGraphDict
from copy import copy, deepcopy
from src_seq.utils import divide_list_into_N, mkdir
from src_seq.tools.timer import Timer
from automata_tools import NFAtoDFA, DFAtoMinimizedDFA
from src_seq.wfa.dfa_from_rule import NFAFromRegex
from src_seq.data import load_slot_dataset, read_rules
from src_seq.wfa.decompose_automata import decompose_automata, decompose_automata_independent, decompose_automata_single
from src_seq.wfa.wfa_utils import check_independent, fix_all_dependent
from src_seq.wfa.wfa_convert import fix_inedge_node
import argparse

timer = Timer()


def get_split_automata(dataset_name, split_group, rules):
    print('GET ALL AUTOMATAS')
    mkdir('../../data/{}/automata'.format(dataset_name))
    if 'ZH' in dataset_name:
        lang = 'zh'
    else:
        lang = 'en'
    assert split_group >= 1
    rule_lists = divide_list_into_N(rules, split_group)
    print('Split the rules in to {} divides'.format(split_group))

    minDFA_objs = []
    total_states = 0
    for splited_rule_i in rule_lists:
        timer.start()

        concatenatedRule = '( ' + " ) | ( ".join(splited_rule_i) + ' )'
        print('\n')
        print(concatenatedRule)
        nfa = NFAFromRegex().buildNFA(concatenatedRule, lang=lang)
        dfa = NFAtoDFA(nfa)
        minDFA = DFAtoMinimizedDFA(dfa)
        minDFA_objs.append(copy(minDFA))
        print('# of states: {}'.format(len(minDFA.states)))
        total_states += len(minDFA.states)
        timer.stop()

    return minDFA_objs, total_states


def create_dataset_automata_slot_multiple_start(
        dataset_name,
        automata_name='none',
        copy_method='none',
        split_group=1,
        rule_name='bio.rules.config',
    ):

    rules = read_rules(dataset_name, rule_name=rule_name)

    print('GETTING RULE FILES')
    data = load_slot_dataset(dataset_name, '../../data/')
    print('TOTAL RULES :{}'.format(len(rules)))

    minDFA_objs, total_states = get_split_automata(dataset_name, split_group, rules)
    if copy_method == 'double':
        L = len(minDFA_objs)
        for i in range(L):
            minDFA_objs.append(deepcopy(minDFA_objs[i]))
        total_states *= 2


    timer.start()
    print('REARRANGE INDEX OF AUTOMATA')
    new_automata = AutomataMultipleStart()

    right_map_idx = 0

    for dfa in minDFA_objs:

        states = list(dfa.states)
        num_states = len(states)
        left_map_idx = right_map_idx
        right_map_idx = left_map_idx + num_states
        used_states = [i for i in range(left_map_idx, right_map_idx)]
        states2idx = {states[i]: used_states[i] for i in range(num_states)}

        # add start states
        new_automata.setstartstate(states2idx[dfa.startstate])

        # add transitions
        for fr_state, to in dfa.transitions.items():
            for to_state, to_edges in to.items():
                for edge in to_edges:
                    word, tag = edge.split('<:>')
                    word = word.lower()
                    tag = tag.lower()
                    assert (tag in data['s2i']) or (tag == 'oo')
                    new_automata.addtransition(states2idx[fr_state], states2idx[to_state], '{}<:>{}'.format(word, tag))

        # add transitions
        new_automata.addfinalstates([states2idx[i] for i in dfa.finalstates])

    rearranged_automata = new_automata.to_dict()

    check_independent(rearranged_automata)

    drawGraph(new_automata, '../../data/{}/automata/{}.{}'.format(dataset_name, automata_name, total_states))

    save_name = 'automata.{}.{}splits.dict'.format(automata_name, split_group,)
    pickle.dump(rearranged_automata, open('../../data/{}/automata/{}'.format(
        dataset_name, save_name), 'wb'))
    timer.stop()

    print('TOTAL STATES: {}'.format(len(rearranged_automata['states'])))

    return rearranged_automata


def get_decomposed_automata(
        dataset,
        automata_name,
        split_group=1,
        kbest=3,
        init='random',
        independent=0,
        decompose=1,
        rule_name='bio.rules.config',
    ):

    rearranged_automata = \
        create_dataset_automata_slot_multiple_start(
            dataset, split_group=split_group,
            copy_method='none',
            automata_name=automata_name,
            rule_name=rule_name)

    if independent in [1, 2]:
        rearranged_automata = fix_inedge_node(rearranged_automata)

    pickle.dump(rearranged_automata, open('../../data/{}/automata/{}.ID{}'.format(
        dataset, automata_name, independent), 'wb'))
    drawGraphDict(rearranged_automata, '../../data/{}/automata/{}.ID{}'.format(
        dataset, automata_name, independent))

    if decompose:
        if independent == 0:
            decompose_automata(
                rearranged_automata,
                dataset,
                automata_name,
                split_group,
                init=init,
                k_best=kbest,
                rule_name=rule_name,
            )
        elif independent == 1:
            decompose_automata_independent(
                rearranged_automata,
                dataset,
                automata_name,
                split_group,
                init=init,
                k_best=kbest,
                rule_name=rule_name,
            )
        else:
            decompose_automata_single(
                rearranged_automata,
                dataset,
                automata_name,
                split_group,
                init=init,
                k_best=kbest,
                rule_name=rule_name,
            )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ATIS-BIO', help="dataset dir")
    parser.add_argument('--automata_name', type=str, default='random', help="automata_name")
    parser.add_argument('--split_group', type=int, default=1000, help="number of splits of automata")
    parser.add_argument('--k_best', type=int, default=5, help="number of bests")
    parser.add_argument('--init', type=str, default='random', help="initialization method")
    parser.add_argument('--independent', type=int, default=2, help='independent type')
    parser.add_argument('--decompose', type=int, default=1)
    parser.add_argument('--rule_name', type=str, default='bio.rules.config')

    args = parser.parse_args()

    assert args.init in ['random', 'svd']
    assert args.independent in [0, 1, 2]
    if args.independent == 0:
        assert args.init == 'random' # 4-order tensor cannot afford svd init

    get_decomposed_automata(
        dataset=args.dataset,
        automata_name=args.automata_name,
        split_group=args.split_group,
        kbest=args.k_best,
        init=args.init,
        independent=args.independent,
        decompose=args.decompose,
        rule_name=args.rule_name
    )
