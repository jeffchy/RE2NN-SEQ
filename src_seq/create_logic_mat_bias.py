import numpy as np


def create_mat_priority_MITR(s2i):
    # create
    mat = np.eye((len(s2i)))

    return mat


def create_mat_priority_MITM(s2i):
    # create
    mat = np.eye((len(s2i)))
    for slot, idx in s2i.items():
        if 'b-' in slot:
            slot_i = 'i-' + slot[2:]
            if slot_i in s2i:
                slot_i_idx = s2i[slot_i]
                mat[slot_i_idx][idx] = -1

    mat[s2i['o']][s2i['i-year']] = -1
    mat[s2i['o']][s2i['b-actor']] = -1

    return mat


def create_mat_priority_SNIPS(s2i):
    # create
    mat = np.eye((len(s2i)))

    for slot, idx in s2i.items():
        if 'b-' in slot:
            slot_i = 'i-' + slot[2:]
            if slot_i in s2i:
                slot_i_idx = s2i[slot_i]
                mat[slot_i_idx][idx] = -1

    mat[s2i['b-playlist_owner']][s2i['b-playlist']] = -1

    return mat


def create_mat_priority_ATIS(s2i):
    # create
    mat = np.eye((len(s2i)))
    for slot, idx in s2i.items():
        if 'b-' in slot:
            slot_i = 'i-' + slot[2:]
            if slot_i in s2i:
                slot_i_idx = s2i[slot_i]
                mat[slot_i_idx][idx] = -1

    return mat


def create_mat_priority_ATIS_ZH(s2i):
    # create
    mat = np.eye((len(s2i)))
    for slot, idx in s2i.items():
        if 'b-' in slot:
            slot_i = 'i-' + slot[2:]
            if slot_i in s2i:
                slot_i_idx = s2i[slot_i]
                mat[slot_i_idx][idx] = -1
    return mat


def create_mat_priority(s2i, args):
    # create
    if 'MITM' in args.dataset:
        return create_mat_priority_MITM(s2i)
    elif 'MITR' in args.dataset:
        return create_mat_priority_MITR(s2i)
    elif 'ATIS-ZH' in args.dataset:
        return create_mat_priority_ATIS_ZH(s2i)
    elif 'ATIS' in args.dataset:
        return create_mat_priority_ATIS(s2i)
    elif 'SNIPS' in args.dataset:
        return create_mat_priority_SNIPS(s2i)
    else:
        raise NotImplementedError()
