import torch
from src_seq.metrics.metrics import eval_seq_token, get_ner_fmeasure
from tqdm import tqdm
from src_seq.utils import unflatten


def val_onehot(dataloader, model, args, o_idx=0, i2s=None, i2t=None, is_cuda=True):
    all_pred_label = []
    all_true_label = []
    data = tqdm(dataloader)
    model.eval()

    if args.method == 'onehot':
        is_cuda = False

    with torch.no_grad():
        for batch in data:

            x = batch['x']
            label = batch['s']
            lengths = batch['l']

            if torch.cuda.is_available() and is_cuda:
                x = x.cuda()
                lengths = lengths.cuda()
                label = label.cuda()

            loss, pred_label, true_label = model.forward_local(x, label, lengths, train=False)

            all_pred_label += list(pred_label.cpu())  # has been flattened
            all_true_label += list(true_label.cpu())
            data.set_postfix_str("{} - Loss: {}".format('EVAL', loss))

    acc, p, r, f = eval_seq_token(seq_label_pred=all_pred_label, seq_label_true=all_true_label, o_idx=o_idx)
    acc_ner, p_ner, r_ner, f_ner, class_res = get_ner_fmeasure(golden_lists=all_true_label, predict_lists=all_pred_label, label_type="BIO", i2s=i2s, all_class=True)

    results = {
        'token-level': [acc, p, r, f],
        'entity-level': [acc_ner, p_ner, r_ner, f_ner, class_res]
    }

    model.train()
    return results


def val_baselines(dataloader, model, args, o_idx=0, i2s=None):
    avg_loss = 0
    all_pred_label = []
    all_true_label = []
    data = tqdm(dataloader)
    model.eval()
    with torch.no_grad():

        for batch in data:

            x = batch['x']
            label = batch['s']
            lengths = batch['l']
            re = batch['re']

            if torch.cuda.is_available():
                x = x.cuda()
                lengths = lengths.cuda()
                label = label.cuda()
                re = re.cuda()

            # # code for global method
            loss, pred_labels, true_label = model(x, label, lengths, re)

            avg_loss += loss.item()

            all_pred_label += list(pred_labels.cpu())  # has been flattened
            all_true_label += list(true_label.cpu())

            data.set_postfix_str("{} - Loss: {}".format('EVAL', loss))

    acc, p, r, f = eval_seq_token(seq_label_pred=all_pred_label, seq_label_true=all_true_label, o_idx=o_idx)
    acc_ner, p_ner, r_ner, f_ner, class_res = get_ner_fmeasure(golden_lists=all_true_label, predict_lists=all_pred_label,
                                                    label_type="BIO", i2s=i2s, all_class=True)

    results = {
        'token-level': [acc, p, r, f],
        'entity-level': [acc_ner, p_ner, r_ner, f_ner, class_res]
    }

    model.train()

    return results


def val_onehot_verbose(dataloader, model, args, o_idx=0, i2s=None, i2t=None):
    all_pred_label = []
    all_true_label = []
    all_pred_label_unflatten = []
    all_true_label_unflatten = []

    data = tqdm(dataloader)
    model.eval()

    if args.method == 'onehot':
        is_cuda = False
    else:
        is_cuda = True

    with torch.no_grad():
        for batch in data:

            x = batch['x']
            label = batch['s']
            lengths = batch['l']

            if torch.cuda.is_available() and is_cuda:
                x = x.cuda()
                lengths = lengths.cuda()
                label = label.cuda()

            loss, pred_label, true_label = model.forward_local(x, label, lengths, train=False)

            all_pred_label += list(pred_label.cpu())  # has been flattened
            all_true_label += list(true_label.cpu())

            all_pred_label_unflatten += unflatten(pred_label.cpu().numpy(), lengths)  # has been flattened
            all_true_label_unflatten += unflatten(true_label.cpu().numpy(), lengths)

            data.set_postfix_str("{} - Loss: {}".format('EVAL', loss))

    acc, p, r, f = eval_seq_token(seq_label_pred=all_pred_label, seq_label_true=all_true_label, o_idx=o_idx)
    acc_ner, p_ner, r_ner, f_ner, class_res = get_ner_fmeasure(golden_lists=all_true_label, predict_lists=all_pred_label, label_type="BIO", i2s=i2s, all_class=True)

    results = {
        'token-level': [acc, p, r, f],
        'entity-level': [acc_ner, p_ner, r_ner, f_ner, class_res],
        'pred_label': all_pred_label_unflatten,
        'true_label': all_true_label_unflatten
    }

    model.train()
    return results


def val_baselines_verbose(dataloader, model, args, o_idx=0, i2s=None):
    avg_loss = 0
    all_pred_label = []
    all_true_label = []
    all_pred_label_unflatten = []
    all_true_label_unflatten = []

    data = tqdm(dataloader)
    model.eval()
    with torch.no_grad():

        for batch in data:

            x = batch['x']
            label = batch['s']
            lengths = batch['l']
            re = batch['re']

            if torch.cuda.is_available():
                x = x.cuda()
                lengths = lengths.cuda()
                label = label.cuda()
                re = re.cuda()

            # # code for global method
            loss, pred_labels, true_label = model(x, label, lengths, re)

            avg_loss += loss.item()

            all_pred_label += list(pred_labels.cpu())  # has been flattened
            all_true_label += list(true_label.cpu())

            all_pred_label_unflatten += unflatten(pred_labels.cpu().numpy(), lengths)  # has been flattened
            all_true_label_unflatten += unflatten(true_label.cpu().numpy(), lengths)

            data.set_postfix_str("{} - Loss: {}".format('EVAL', loss))

    acc, p, r, f = eval_seq_token(seq_label_pred=all_pred_label, seq_label_true=all_true_label, o_idx=o_idx)
    acc_ner, p_ner, r_ner, f_ner, class_res = get_ner_fmeasure(golden_lists=all_true_label, predict_lists=all_pred_label,
                                                    label_type="BIO", i2s=i2s, all_class=True)

    results = {
        'token-level': [acc, p, r, f],
        'entity-level': [acc_ner, p_ner, r_ner, f_ner, class_res],
        'pred_label': all_pred_label_unflatten,
        'true_label': all_true_label_unflatten
    }

    model.train()

    return results