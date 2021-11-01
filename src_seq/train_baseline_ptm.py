import torch
from src_seq.utils import Logger, len_stats, pad_dataset_1, set_seed
from src_seq.data import load_slot_dataset, SlotBatchDatasetPTM
from torch.utils.data import DataLoader
from src_seq.baselines.bert_simple_sf import Bert_SF
from tqdm import tqdm
from src_seq.metrics.metrics import eval_seq_token, get_ner_fmeasure
from src_seq.val_ptm import val_onehot_ptm
from src_seq.tools.saver import save_model_and_log
from src_seq.tools.printer import print_and_log_results, Best_Model_Recorder


def train_slot_neural_softmax_and_marryup_baselines_ptm(args):

    # Settings
    logger = Logger()
    dataset_name = args.dataset
    seq_max_len = args.seq_max_len
    train_portion = args.train_portion
    bz = args.bz
    random_embed = bool(args.random_embed)
    optimizer = args.optimizer
    seed = args.seed
    embed_dim = args.embed_dim

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

    slot_data_train = SlotBatchDatasetPTM(train_query, train_lengths, slot_train,
                                              args, i2t, train_portion, dset='train')
    slot_data_dev = SlotBatchDatasetPTM(dev_query, dev_lengths, slot_dev,
                                            args, i2t, train_portion, dset='dev')
    slot_data_test = SlotBatchDatasetPTM(test_query, test_lengths, slot_test,
                                             args, i2t, dset='test')

    print('Train Samples: ', len(slot_data_train))
    print('Dev Samples: ', len(slot_data_dev))
    print('Test Samples: ', len(slot_data_test))

    slot_dataloader_train = DataLoader(slot_data_train, batch_size=bz)
    slot_dataloader_dev = DataLoader(slot_data_dev, batch_size=bz)
    slot_dataloader_test = DataLoader(slot_data_test, batch_size=bz)

    label_size = len(s2i)

    # for padding
    model = Bert_SF(args=args,
                    label_size=label_size,
                    o_idx=s2i['o'])

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('ALL TRAINABLE PARAMETERS: {}'.format(pytorch_total_params))

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD([
            {'params': model.linear.parameters()},
            {'params': model.crf.parameters()},
            {'params': model.embed.bert.parameters(), 'lr': args.lr/args.bert_lr_down_factor},
        ], lr=args.lr, weight_decay=0)

    if args.optimizer == 'ADAM':
        optimizer = torch.optim.Adam([
            {'params': model.linear.parameters()},
            {'params': model.crf.parameters()},
            {'params': model.embed.bert.parameters(), 'lr': args.lr/args.bert_lr_down_factor},
        ], lr=args.lr, weight_decay=0)

    if torch.cuda.is_available():
        model = model.cuda()

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
    o_idx = s2i['o']

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
            re_tags = batch['re']

            if torch.cuda.is_available():
                x_bert = x_bert.cuda()
                x = x.cuda()
                attend_mask = attend_mask.cuda()
                valid_mask = valid_mask.cuda()
                lengths = lengths.cuda()
                label = label.cuda()
                re_tags = re_tags.cuda()

            # # code for global method
            loss, pred_label, true_label = model(x, x_bert, attend_mask, valid_mask,
                                                 lengths, label, train=True, re_tags=re_tags)

            loss.backward()
            optimizer.step()
            model.update()
            avg_loss += loss.item()

            all_pred_label += list(pred_label.cpu())
            all_true_label += list(true_label.cpu())

            pbar_train.set_postfix_str("{} - Epoch: {} - Loss: {}".format('TRAIN', epoch, loss))

        avg_loss = avg_loss / len(slot_data_train)
        print("{} Epoch: {} | LOSS: {}".format('TRAIN', epoch, avg_loss))
        logger.add("{} Epoch: {} | LOSS: {}".format('TRAIN', epoch, avg_loss))

        results_train = dict()
        results_train['token-level'] = list(eval_seq_token(seq_label_pred=all_pred_label, seq_label_true=all_true_label,  o_idx=s2i['o']))
        results_train['entity-level'] = list(get_ner_fmeasure(golden_lists=all_true_label, predict_lists=all_pred_label, i2s=i2s))
        print_and_log_results(logger, results_train, epoch, 'TRAIN')

        if args.use_unlabel:
            results_dev = results_train # we dont have dev set in unlabel setting
        else:
            results_dev = val_onehot_ptm(slot_dataloader_dev, model, args, s2i['o'], i2s)
            print_and_log_results(logger, results_dev, epoch, 'DEV')

        results_test = val_onehot_ptm(slot_dataloader_test, model, args, s2i['o'], i2s)

        best_recorder.update_and_record(results_train, results_dev, results_test, model.state_dict())

    save_model_and_log(logger, best_recorder, args)