import torch


def split_dev(dev_dataset, dev_slot, dev_dataset_re_results, dev_lengths, s2i):
    assert len(dev_dataset) == len(dev_dataset_re_results)

    re_idx = []
    no_re_idx = []
    for sent_re_res_id in range(len(dev_dataset_re_results)):
        for word_re_res_id in range(dev_lengths[sent_re_res_id]):
            if dev_dataset_re_results[sent_re_res_id][word_re_res_id] != s2i['o']:
                re_idx.append(sent_re_res_id)
                break
        else:
            no_re_idx.append(sent_re_res_id)

    dev_re = [dev_dataset[i] for i in re_idx]
    dev_lengths_re = [dev_lengths[i] for i in re_idx]
    dev_slot_re = [dev_slot[i] for i in re_idx]
    dev_no_re = [dev_dataset[i] for i in no_re_idx]
    dev_lengths_no_re = [dev_lengths[i] for i in no_re_idx]
    dev_slot_no_re = [dev_slot[i] for i in no_re_idx]

    return dev_re, dev_slot_re, dev_lengths_re, dev_no_re, dev_slot_no_re, dev_lengths_no_re


def split_dev_marryup(dev_dataset, dev_slot, dev_dataset_re_results, dev_re_score, dev_lengths, s2i):
    assert len(dev_dataset) == len(dev_dataset_re_results)

    re_idx = []
    no_re_idx = []
    for sent_re_res_id in range(len(dev_dataset_re_results)):
        for word_re_res_id in range(dev_lengths[sent_re_res_id]):
            if dev_dataset_re_results[sent_re_res_id][word_re_res_id] != s2i['o']:
                re_idx.append(sent_re_res_id)
                break
        else:
            no_re_idx.append(sent_re_res_id)

    dev_re = [dev_dataset[i] for i in re_idx]
    dev_lengths_re = [dev_lengths[i] for i in re_idx]
    dev_slot_re = [dev_slot[i] for i in re_idx]
    dev_result_re = torch.stack([dev_re_score[i] for i in re_idx], dim=0)
    dev_no_re = [dev_dataset[i] for i in no_re_idx]
    dev_lengths_no_re = [dev_lengths[i] for i in no_re_idx]
    dev_slot_no_re = [dev_slot[i] for i in no_re_idx]
    temp = [dev_re_score[i] for i in no_re_idx]
    if len(dev_no_re) > 0:
        dev_result_no_re = torch.stack([dev_re_score[i] for i in no_re_idx])
    else:
        dev_result_no_re = torch.Tensor([])

    return dev_re, dev_slot_re, dev_result_re, dev_lengths_re, dev_no_re, dev_slot_no_re, dev_result_no_re, dev_lengths_no_re
