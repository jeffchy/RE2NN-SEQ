import numpy as np
import random
import torch
import os
import datetime, time
import pickle


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def len_stats(query):
    max_len = 0
    avg_len = 0
    for q in query:
        max_len = max(len(q), max_len)
        avg_len += len(q)
    avg_len /= len(query)

    print("max_len: {}, avg_len: {}".format(max_len, avg_len))


def pad_dataset_1(query, seq_max_len, pad_idx):

    lengths = []
    new_query = []
    new_query_inverse = []
    for q in query:
        length = len(q)

        if length <= 0:
            continue

        q_inverse = q[::-1]

        if length > seq_max_len:
            q = q[: seq_max_len]
            q_inverse = q_inverse[: seq_max_len]
            length = seq_max_len
        else:
            remain = seq_max_len - length
            remain_arr = np.repeat(pad_idx, remain)
            q = np.concatenate((q, remain_arr))
            q_inverse = np.concatenate((q_inverse, remain_arr))
            assert len(q) == seq_max_len

        new_query.append(q)
        new_query_inverse.append(q_inverse)
        lengths.append(length)

    return new_query, new_query_inverse, lengths


def pad_dataset(query, config, pad_idx):

    lengths = []
    new_query = []
    new_query_inverse = []
    for q in query:
        length = len(q)
        q_inverse = q[::-1]

        if length > config.seq_max_len:
            q = q[: config.seq_max_len]
            q_inverse = q_inverse[: config.seq_max_len]
            length = config.seq_max_len
        else:
            remain = config.seq_max_len - length
            remain_arr = np.repeat(pad_idx, remain)
            q = np.concatenate((q, remain_arr))
            q_inverse = np.concatenate((q_inverse, remain_arr))
            assert len(q) == config.seq_max_len

        new_query.append(q)
        new_query_inverse.append(q_inverse)
        lengths.append(length)

    return new_query, new_query_inverse, lengths


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def create_datetime_str():
    datetime_dt = datetime.datetime.today()
    datetime_str = datetime_dt.strftime("%m%d%H%M%S")
    datetime_str = datetime_str + '-' + str(time.time())
    return datetime_str


class args():
    def __init__(self, data):
        self.data = data
        for k, v in data.items():
            setattr(self, k, v)


class Args():
    def __init__(self, data):
        self.data = data
        for k, v in data.items():
            setattr(self, k, v)


class Logger():
    def __init__(self):
        self.record = [] # recored strings

    def add(self, string):
        assert type(string) == str
        self.record.append(string+' \n')

    def save(self, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            f.writelines(self.record)


def flatten_batch(batch, length):
    B, L = batch.size()
    flatten = []
    for i in range(B):
        flatten += list(batch[i, :length[i]])
    return flatten


def get_length_mask(length, max_len=None, ):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    borrowed from https://discuss.pytorch.org/t/how-to-generate-variable-length-mask/23397/3
    """
    assert len(length.shape) == 1
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)

    return mask.bool()


def load_pkl(path):
    print(path)
    dicts = pickle.load(open(path, 'rb'))
    return dicts


def flatten(input, length):
    """
    :param input: B x L x ?
    :param length: B
    :return:
    """

    B, L = input.size()[0], input.size()[1]
    # fraction = 1
    flattened = torch.cat([input[i,:length[i]] for i in range(B)], dim=0)

    return flattened


def unflatten(input, length):
    """
    :param input: K ?
    :param length: B
    :return: list of list
    """

    unflattened = []
    idx = 0
    for i in length:
        unflattened.append(input[idx:idx+i])
        idx += i

    return unflattened


def reverse(input, lengths):
    B = input.size()[0] # B x L
    reversed_input = input.clone() # clone is ok because input does not requires grad
    for i in range(B):
        reversed_input[i, :lengths[i]] = input[i, :lengths[i]].flip([0])

    return reversed_input


def _maxmul(hidden, transition):
    temp = torch.einsum('bs,bsj->bsj', hidden, transition)
    max_val, _ = torch.max(temp, dim=1)
    return max_val


def _matmul(hidden, transition):
    return torch.einsum('bs,bsj->bj', hidden, transition)


def get_average(M, normalize_type):
    """
    :param M:
    :param normalize_type:
    :return:
    Get the averaged norm
    """
    assert normalize_type in ['l1', 'l2', 'l1-rank', 'l2-rank']

    if normalize_type == 'l1':
        temp = np.linalg.norm(M, 1)
        eles = M.size
    elif normalize_type == 'l2':
        temp = np.linalg.norm(M, 2)
        eles = M.size
    elif normalize_type == 'l1-rank':
        temp = np.linalg.norm(M, 1, axis=0)
        eles = M.shape[0]
    elif normalize_type == 'l2-rank':
        temp = np.linalg.norm(M, 2, axis=0)
        eles = M.shape[0]

    return temp / eles


def divide_list_into_N(list_obj, N):
    L = len(list_obj)
    avg_n = L / N
    splited_lists = []

    for i in range(1, N+1):
        l = int((i-1)*avg_n)
        r = int(i*avg_n)
        if r > l:
            splited_lists.append(list_obj[l: r])

    return splited_lists


def even_select_from_portion(L, portion, seed=0):
    final_nums = int(L * portion)
    interval = 1 / portion
    idxs = [int(i*interval)+seed for i in range(final_nums)]
    return np.array(idxs)


def even_select_from_total_number(L, N, seed=0):
    # assert L >= N
    if 0 < N < L:
        # portion = N / L
        # interval = 1 / portion
        # idxs = [int(i*interval)+seed for i in range(N)]

        return np.random.choice(L, N, replace=False)

    elif N >= L:
        idxs = [i for i in range(L)]
    else:
        idxs = []
    return np.array(idxs)


def bilinear(alpha, trans, beta):

    temp = trans * alpha * beta
    res = torch.sum(temp, dim=(1,2)) # SUM
    # B, S1, S2 = temp.size()
    # res, _ = torch.max(temp.view(B, -1), dim=1) # MAX
    return res


def add_random_noise(obj, amp=0.00001):
    return obj + torch.rand_like(obj) * amp


def add_small_constant(obj, amp=0.0001, rand=False):
    if rand:
        return obj + amp * torch.rand_like(obj)
    else:
        return obj + amp


def add_max_val(obj, amp=0.0001):
    return torch.max(obj, torch.tensor(amp).to(obj.device))

def select_by_vec(obj, select_vec):
    if len(select_vec) == 1:
        return obj[:select_vec[0]],\
               obj[select_vec[0]:]
    elif len(select_vec) == 2:
        return obj[:select_vec[0], :select_vec[1]],\
               obj[select_vec[0]:, select_vec[1]:]
    elif len(select_vec) == 3:
        return obj[:select_vec[0], :select_vec[1], :select_vec[2]],\
               obj[select_vec[0]:, select_vec[1]:, select_vec[2]:]
    elif len(select_vec) == 4:
        return obj[:select_vec[0], :select_vec[1], :select_vec[2], :select_vec[3]],\
               obj[select_vec[0]:, select_vec[1]:, select_vec[2]:, select_vec[3]:]
    else:
        raise NotImplementedError()


def xavier_normal(obj):
    """
    :param matrix obj in numpy:
    :return xavier random normal with same size of obj:
    """
    std = np.sqrt(2. / np.sum(obj.shape))
    return np.random.normal(loc=0., scale=std, size=obj.shape)


