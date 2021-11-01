# modified from
# https://github.com/jiesutd/NCRFpp/blob/master/utils/metric.py

from __future__ import print_function
import sys

def eval_seq_token(seq_label_pred, seq_label_true, o_idx=0):
    """
    :param seq_label_pred: B x L
    :param seq_label_true: B x L
    :param seq_len: B
    :return:
    """
    assert len(seq_label_pred) == len(seq_label_true)

    correct = 0
    tp = 0
    fp = 0
    fn = 0

    for i in range(len(seq_label_pred)):
        sp = seq_label_pred[i]
        st = seq_label_true[i]

        if sp == st:
            correct += 1
            if sp != o_idx:
                tp += 1
        else:
            if sp != o_idx:
                fp += 1
            if st != o_idx:
                fn += 1


    all_tokens = len(seq_label_pred)
    accuracy = correct / all_tokens
    precision = tp / (tp + fp) if (tp + fp != 0) else 0
    recall = tp / (tp + fn) if (tp + fn != 0) else 0
    f1 = 2 * precision * recall / (precision + recall) if  (precision + recall != 0) else 0

    return accuracy, precision, recall, f1

def get_matrix_for_each_class(pred_matrix, gold_matrix):
    all_dict = {}
    for pred in pred_matrix:
        _, C = pred.split(']')
        if C not in all_dict:
            all_dict[C] = [[pred], []] # pred, gold
        else:
            all_dict[C][0].append(pred)

    for gold in gold_matrix:
        _, C = gold.split(']')
        if C not in all_dict:
            all_dict[C] = [[], [gold]] # pred, gold
        else:
            all_dict[C][1].append(gold)

    return all_dict

def get_results_from_mat(pred_matrix, gold_matrix):

    right_ner = list(set(gold_matrix).intersection(set(pred_matrix)))
    right_num = len(right_ner)
    golden_num = len(gold_matrix)
    predict_num = len(pred_matrix)

    if predict_num == 0:
        precision = -1
    else:
        precision =  (right_num+0.0)/predict_num
    if golden_num == 0:
        recall = -1
    else:
        recall = (right_num+0.0)/golden_num
    if (precision == -1) or (recall == -1) or (precision+recall) <= 0.:
        f_measure = -1
    else:
        f_measure = 2*precision*recall/(precision+recall)

    return precision, recall, f_measure

def get_all_class_measure(all_dict):
    # print('P, R, F for labels')
    results_dict = {}
    for k, v in all_dict.items():
        p,r,f = get_results_from_mat(v[0], v[1])
        results_dict[k] = [p, r, f]
        # print('{}:{},{},{}'.format(k, p, r, f), end=' ')
    return results_dict



## input as sentence level labels
def get_ner_fmeasure(golden_lists, predict_lists, label_type="BIO", i2s=None, all_class=False):

    golden_full = []
    predict_full = []
    right_full = []
    right_tag = 0
    all_tag = 0

    # convert to slots
    golden_lists = [i2s[int(m)] for m in golden_lists]
    predict_lists = [i2s[int(m)] for m in predict_lists]

    golden_list = golden_lists
    predict_list = predict_lists
    for idy in range(len(golden_list)):
        if golden_list[idy] == predict_list[idy]:
            right_tag += 1
    all_tag += len(golden_list)
    if label_type == "BMES" or label_type == "BIOES":
        gold_matrix = get_ner_BMES(golden_list)
        pred_matrix = get_ner_BMES(predict_list)
    else:
        gold_matrix = get_ner_BIO(golden_list)
        pred_matrix = get_ner_BIO(predict_list)

    right_ner = list(set(gold_matrix).intersection(set(pred_matrix)))
    precision, recall, f_measure = get_results_from_mat(pred_matrix, gold_matrix)
    accuracy = (right_tag+0.0)/all_tag

    class_results_dict = None
    if all_class:
        class_results_dict = get_all_class_measure(all_dict=get_matrix_for_each_class(pred_matrix, gold_matrix))

    return accuracy, precision, recall, f_measure, class_results_dict


def reverse_style(input_string):
    target_position = input_string.index('[')
    input_len = len(input_string)
    output_string = input_string[target_position:input_len] + input_string[0:target_position]
    return output_string


def get_ner_BMES(label_list):

    list_len = len(label_list)
    begin_label = 'B-'
    end_label = 'E-'
    single_label = 'S-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i-1))
            whole_tag = current_label.replace(begin_label,"",1) +'[' +str(i)
            index_tag = current_label.replace(begin_label,"",1)

        elif single_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i-1))
            whole_tag = current_label.replace(single_label,"",1) +'[' +str(i)
            tag_list.append(whole_tag)
            whole_tag = ""
            index_tag = ""
        elif end_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag +',' + str(i))
            whole_tag = ''
            index_tag = ''
        else:
            continue
    if (whole_tag != '')&(index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if  len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i]+ ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    # print stand_matrix
    return stand_matrix


def get_ner_BIO(label_list):
    # list_len = len(word_list)
    # assert(list_len == len(label_list)), "word list size unmatch with label list"
    list_len = len(label_list)
    begin_label = 'B-'
    inside_label = 'I-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        # wordlabel = word_list[i]
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag == '':
                whole_tag = current_label.replace(begin_label,"",1) + '[' + str(i)
                index_tag = current_label.replace(begin_label,"",1)
            else:
                tag_list.append(whole_tag + ',' + str(i-1))
                whole_tag = current_label.replace(begin_label,"",1) + '[' + str(i)
                index_tag = current_label.replace(begin_label,"",1)

        elif inside_label in current_label:
            if current_label.replace(inside_label,"",1) == index_tag:
                whole_tag = whole_tag
            else:
                if (whole_tag != '')&(index_tag != ''):
                    tag_list.append(whole_tag +',' + str(i-1))
                whole_tag = ''
                index_tag = ''
        else:
            if (whole_tag != '')&(index_tag != ''):
                tag_list.append(whole_tag +',' + str(i-1))
            whole_tag = ''
            index_tag = ''

    if (whole_tag != '')&(index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if  len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i] + ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    return stand_matrix



def readSentence(input_file):
    in_lines = open(input_file,'r').readlines()
    sentences = []
    labels = []
    sentence = []
    label = []
    for line in in_lines:
        if len(line) < 2:
            sentences.append(sentence)
            labels.append(label)
            sentence = []
            label = []
        else:
            pair = line.strip('\n').split(' ')
            sentence.append(pair[0])
            label.append(pair[-1])
    return sentences,labels




