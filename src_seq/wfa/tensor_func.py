import tensorly as tl
from tensorly.decomposition import parafac
import numpy as np
import time


def tensor3_to_factors(tensor, rank, n_iter_max=50, init='svd', verbose=10, random_state=1):
    assert init in ['random', 'svd']
    tensor = tl.tensor(tensor)
    factors, rec_errors = parafac(tensor, rank=rank, random_state=random_state, normalize_factors=False, verbose=verbose,
                                    n_iter_max=n_iter_max, tol=1e-4, init=init, return_errors=True)

    return factors[1][0], factors[1][1], factors[1][2], rec_errors


def tensor4_to_factors(tensor, rank, n_iter_max=60, init='svd', verbose=10, random_state=1):
    assert init in ['random', 'svd']

    tensor = tl.tensor(tensor)

    factors, rec_errors = parafac(tensor, rank=rank, random_state=random_state, normalize_factors=False, verbose=verbose,
                                        n_iter_max=n_iter_max, tol=1e-6, init=init, return_errors=True)

    if rec_errors[-1] < 0:
        raise ValueError

    return factors[1][0], factors[1][1], factors[1][2], factors[1][3], rec_errors


def recover_tensor_from_factors(factors):

    recovered = tl.cp_to_tensor(factors)
    return recovered



def decompose_tensor_split_slot_4d(
    language_tensor, language, word2idx, rank, random_state=1, n_iter_max=20, init='svd', verbose=10,
):
    language_tensor_squashed = language_tensor[np.array([word2idx[i] for i in language])]

    print('SQUASHED TENSOR SIZE: {}'.format(language_tensor_squashed.shape))
    time_start = time.time()
    V_embed, C_embed, S1, S2, rec_error = tensor4_to_factors(language_tensor_squashed,
                                                             rank=rank,
                                                             n_iter_max=n_iter_max,
                                                             verbose=verbose,
                                                             random_state=random_state,
                                                             init=init,)
    time_end = time.time()
    print('time cost', time_end - time_start, 's')

    vocab, labels, state, _ = language_tensor.shape

    V_embed_split = np.zeros((vocab, rank))

    for i in range(len(language)):
        idx = word2idx[language[i]]
        V_embed_split[idx] = V_embed[i]

    return V_embed_split, C_embed,  S1, S2, rec_error


def decompose_tensor_split_slot_3d(
    the_tensor, rank, random_state=1, n_iter_max=20, init='svd', verbose=10,
):

    print('SQUASHED TENSOR SIZE: {}'.format(the_tensor.shape))
    time_start = time.time()
    C_embed, S1, S2, rec_error = tensor3_to_factors(the_tensor, rank=rank,
                                                    n_iter_max=n_iter_max, verbose=verbose,
                                                    random_state=random_state, init=init,)
    time_end = time.time()
    print('time cost', time_end - time_start, 's')

    return C_embed,  S1, S2, rec_error


def decompose_tensor_split_slot_3d_language(
    language_tensor, language, word2idx, rank, random_state=1, n_iter_max=20, init='svd', verbose=10,
):
    language_tensor_squashed = language_tensor[np.array([word2idx[i] for i in language])]

    print('SQUASHED TENSOR SIZE: {}'.format(language_tensor_squashed.shape))
    time_start = time.time()
    V_embed, S1, S2, rec_error = tensor3_to_factors(language_tensor_squashed, rank=rank,
                                                    n_iter_max=n_iter_max, verbose=verbose,
                                                    random_state=random_state, init=init,)

    time_end = time.time()
    print('time cost', time_end - time_start, 's')

    vocab, _, state = language_tensor.shape

    V_embed_split = np.zeros((vocab, rank))

    for i in range(len(language)):
        idx = word2idx[language[i]]
        V_embed_split[idx] = V_embed[i]

    return V_embed_split, S1, S2, rec_error