import torch
from src_seq.utils import Logger, len_stats, pad_dataset_1, set_seed
from src_seq.data import load_slot_dataset, SlotBatchDatasetPTM
from src_seq.ptm.bert_utils import load_bert_embed
from torch.utils.data import DataLoader
from tqdm import tqdm
from src_seq.metrics.metrics import eval_seq_token, get_ner_fmeasure
from src_seq.val_ptm import val_onehot_ptm
from src_seq.tools.saver import save_model_and_log
from src_seq.tools.printer import Best_Model_Recorder, print_and_log_results
from src_seq.farnn.model_decompose_single_with_bert import FARNN_S_bert
from src_seq.init_params import get_init_params_seq_independent, get_init_params_seq, \
                                get_init_params_seq_independent_single, get_init_random_params


def train_slot_decompose_ptm(args):

    # Settings
    logger = Logger()
    dataset_name = args.dataset
    seq_max_len = args.seq_max_len
    train_portion = args.train_portion
    bz = args.bz
    optimizer = args.optimizer
    seed = args.seed

    set_seed(seed)

    dset = load_slot_dataset(dataset_name)
    t2i, i2t, s2i, i2s = dset['t2i'], dset['i2t'], dset['s2i'], dset['i2s']
    query_train, slot_train = dset['query_train'], dset['intent_train']
    query_dev, slot_dev = dset['query_dev'], dset['intent_dev']
    query_test, slot_test = dset['query_test'], dset['intent_test']

    len_stats(query_train)
    len_stats(query_dev)
    len_stats(query_test)
    # extend the padding
    # add pad <pad> to the last of vocab
    i2t[len(i2t)] = '<pad>'
    t2i['<pad>'] = len(i2t) - 1

    train_query, _, train_lengths = pad_dataset_1(query_train, seq_max_len, t2i['<pad>'])
    dev_query, _, dev_lengths = pad_dataset_1(query_dev, seq_max_len, t2i['<pad>'])
    test_query, _, test_lengths = pad_dataset_1(query_test, seq_max_len, t2i['<pad>'])
    slot_train, _, _ = pad_dataset_1(slot_train, seq_max_len, s2i['o'])
    slot_dev, _, _ = pad_dataset_1(slot_dev, seq_max_len, s2i['o'])
    slot_test, _, _ = pad_dataset_1(slot_test, seq_max_len, s2i['o'])

    slot_data_train = SlotBatchDatasetPTM(train_query, train_lengths, slot_train, args, i2t,
                                       portion=train_portion, dset='train',)
    slot_data_dev = SlotBatchDatasetPTM(dev_query, dev_lengths, slot_dev, args, i2t,
                                     portion=train_portion, dset='dev',)
    slot_data_test = SlotBatchDatasetPTM(test_query, test_lengths, slot_test,
                                      args, i2t, dset='test',)

    print('Train Samples: ', len(slot_data_train))
    print('Dev Samples: ', len(slot_data_dev))
    print('Test Samples: ', len(slot_data_test))

    slot_dataloader_train = DataLoader(slot_data_train, batch_size=bz)
    slot_dataloader_dev = DataLoader(slot_data_dev, batch_size=bz)
    slot_dataloader_test = DataLoader(slot_data_test, batch_size=bz)

    vocab_size = len(t2i)
    label_size = len(s2i)

    # TODO: we currently not support CE1 for 4-order method
    if args.local_loss_func == 'CE1':
       assert args.independent != 0

    V_embed_extend, S1, S2, pretrain_embed_extend, wildcard_mat, wildcard_output_vector, \
    final_vector, start_vector, priority_mat, C_output_mat, bert_embed \
    = get_init_params_seq_independent_single(args, s2i, t2i)

    model = FARNN_S_bert(
        V=V_embed_extend,
        S1=S1,
        S2=S2,
        C_output_mat=C_output_mat,
        wildcard_mat=wildcard_mat,
        wildcard_output_vector=wildcard_output_vector,
        final_vector=final_vector,
        start_vector=start_vector,
        static_embed=bert_embed,
        priority_mat=priority_mat,
        args=args,
        o_idx=s2i['o'])

    n_state, rank = S1.shape

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD([
            {'params': model.slot_filler.parameters()},
            {'params': [model.embed.embed_r_generalized,
                        model.embed.V_embed,
                        model.embed.beta_vec,]},
            {'params': model.embed.embed.bert.parameters(), 'lr': args.lr/args.bert_lr_down_factor},
        ], lr=args.lr, weight_decay=0)

    if args.optimizer == 'ADAM':
        optimizer = torch.optim.Adam([
            {'params': model.slot_filler.parameters()},
            {'params': [model.embed.embed_r_generalized,
                        model.embed.V_embed,
                        model.embed.beta_vec,]},
            {'params': model.embed.embed.bert.parameters(), 'lr': args.lr/args.bert_lr_down_factor},
        ], lr=args.lr, weight_decay=0)

    if torch.cuda.is_available():
        model = model.cuda()

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('ALL TRAINABLE PARAMETERS: {}'.format(pytorch_total_params))

    results_train = val_onehot_ptm(slot_dataloader_train, model, args, s2i['o'], i2s)
    print_and_log_results(logger, results_train, 'INIT', 'TRAIN')

    results_dev = val_onehot_ptm(slot_dataloader_dev, model, args, s2i['o'], i2s)
    print_and_log_results(logger, results_dev, 'INIT', 'DEV')

    results_test = val_onehot_ptm(slot_dataloader_test, model, args, s2i['o'], i2s)
    print_and_log_results(logger, results_test, 'INIT', 'TEST')

    best_recorder = Best_Model_Recorder(selector='f', level=args.select_level,
                                        init_results_train=results_train,
                                        init_results_dev=results_dev,
                                        init_results_test=results_test,
                                        save_model=bool(args.save_model))

    for epoch in range(1, args.epoch + 1):

        all_pred_label = []
        all_true_label = []

        avg_loss = 0

        pbar_train = tqdm(slot_dataloader_train)
        pbar_train.set_description("TRAIN EPOCH {}".format(epoch))

        for batch in pbar_train:

            optimizer.zero_grad()
            x_bert = batch['x_bert']
            x = batch['x']
            attend_mask = batch['attend_mask']
            valid_mask = batch['valid_mask']
            label = batch['s']
            lengths = batch['l']
            if args.marryup_type in ['kd', 'pr']:
                re_tags = batch['re']

            if torch.cuda.is_available():
                x_bert = x_bert.cuda()
                x = x.cuda()
                attend_mask = attend_mask.cuda()
                valid_mask = valid_mask.cuda()
                lengths = lengths.cuda()
                label = label.cuda()
                if args.marryup_type in ['kd', 'pr']:
                    re_tags = re_tags.cuda()

            if args.marryup_type in ['kd', 'pr']:
                loss, pred_label, true_label = model(x, x_bert, attend_mask, valid_mask,
                                                     lengths, label, train=True, re_tags=re_tags)
            else:
                loss, pred_label, true_label = model(x, x_bert, attend_mask, valid_mask,
                                                     lengths, label, train=True, re_tags=None)

            loss.backward()
            optimizer.step()
            avg_loss += loss.item()

            all_pred_label += list(pred_label.cpu()) # has been flattened
            all_true_label += list(true_label.cpu())

            pbar_train.set_postfix_str("{} - Epoch: {} - Loss: {}".format(
                                        'TRAIN', epoch, avg_loss))

        avg_loss = avg_loss / len(slot_data_train)
        print("{} Epoch: {} | LOSS: {}".format('TRAIN', epoch, avg_loss))
        logger.add("{} Epoch: {} | LOSS: {}".format('TRAIN', epoch, avg_loss))

        results_train = dict()
        results_train['token-level'] = list(eval_seq_token(seq_label_pred=all_pred_label, seq_label_true=all_true_label,
                                                      o_idx=s2i['o']))
        results_train['entity-level'] = list(get_ner_fmeasure(golden_lists=all_true_label, predict_lists=all_pred_label,
                                                         i2s=i2s))
        print_and_log_results(logger, results_train, epoch, 'TRAIN')

        results_dev = val_onehot_ptm(slot_dataloader_dev, model, args, s2i['o'], i2s)
        print_and_log_results(logger, results_dev, epoch, 'DEV')

        results_test = val_onehot_ptm(slot_dataloader_test, model, args, s2i['o'], i2s)

        best_recorder.update_and_record(results_train, results_dev, results_test, model.state_dict())
        print()

    print(best_recorder.best_dev_test_results)

    save_model_and_log(logger, best_recorder, args)