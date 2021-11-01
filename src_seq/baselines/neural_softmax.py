import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from src_seq.utils import flatten, get_length_mask
from src_seq.baselines.crf import CRF
from src_seq.baselines.KD import KD_loss, PR_loss


class SlotNeuralSoftmax(nn.Module):
    def __init__(self, pretrained_embed, args, label_size, o_idx):
        super(SlotNeuralSoftmax, self).__init__()

        self.args = args
        self.bidirection = bool(args.bidirection)
        self.is_cuda = torch.cuda.is_available()
        self.V, self.D = pretrained_embed.shape
        self.rnn = args.rnn
        self.o_idx = o_idx

        self.margin = torch.tensor(args.margin).float().cuda() if self.is_cuda else torch.tensor(
            args.margin).float()
        self.threshold = torch.tensor(args.threshold).float().cuda() if self.is_cuda else torch.tensor(
            args.threshold).float()
        self.zero = torch.tensor(0).float().cuda() if self.is_cuda else torch.tensor(0).float()
        self.epsilon = torch.tensor(1e-5).cuda() if self.is_cuda else torch.tensor(1e-5)

        self.rnn_hidden_dim = args.rnn_hidden_dim // 2 if self.bidirection else args.rnn_hidden_dim

        self.use_crf = bool(args.use_crf)
        self.label_size = label_size + 2 if self.use_crf else label_size
        self.crf = CRF(label_size, self.is_cuda)
        self.re_tag_embed = nn.Parameter(torch.randn((self.label_size, args.re_tag_dim)), requires_grad=True)
        self.logits_weights = nn.Parameter(torch.randn(self.label_size), requires_grad=True)  # C,
        self.input_dim = self.D + self.args.re_tag_dim if self.args.marryup_type in ['all', 'input'] else self.D
        self.t = 1

        if self.args.local_loss_func == 'CE':
            self.loss = nn.CrossEntropyLoss()
        elif self.args.local_loss_func == 'ML':
            self.loss = nn.MultiMarginLoss(margin=self.margin)
        else:
            raise NotImplementedError()

        if args.rnn == 'RNN':
            self.rnn = nn.RNN(input_size=self.input_dim,
                              hidden_size=self.rnn_hidden_dim,
                              num_layers=1,
                              batch_first=True,
                              bidirectional=self.bidirection,
                              bias=False)

        elif args.rnn == 'LSTM':
            self.rnn = nn.LSTM(input_size=self.input_dim,
                               hidden_size=self.rnn_hidden_dim,
                               num_layers=1,
                               batch_first=True,
                               bidirectional=self.bidirection,
                               bias=False)

        elif args.rnn == 'GRU':
            self.rnn = nn.GRU(input_size=self.input_dim,
                              hidden_size=self.rnn_hidden_dim,
                              num_layers=1,
                              batch_first=True,
                              bidirectional=self.bidirection,
                              bias=False)

        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(pretrained_embed).float(),
                                                      freeze=(not args.train_word_embed))  # V x D

        self.directions = 2 if self.bidirection else 1
        self.linear = nn.Linear(self.directions * self.rnn_hidden_dim, self.label_size)

    def forward(self, input, label, lengths, re_tags, train=True):
        # re_tags B x Label
        input = self.embedding(input)  # B x L x D
        if self.args.use_crf:
            B, L, C = re_tags.size()
            re_tags = torch.cat([re_tags, torch.zeros(B, L, 2).to(re_tags.device)], dim=2)

        if self.args.marryup_type in ['input', 'all']:
            a = (re_tags > 0).sum()
            re_tag_embed_input = torch.einsum('blc,cd->bld', re_tags, self.re_tag_embed) / torch.max(
                                                                        re_tags.sum(-1).unsqueeze(-1), self.epsilon)
            input = torch.cat([input, re_tag_embed_input], dim=2)

        pack_padded_seq_input = pack_padded_sequence(input, lengths, batch_first=True, enforce_sorted=False)

        B, L, D = input.size()

        if self.args.rnn in ['RNN', 'GRU']:
            output_packed, hn = self.rnn(pack_padded_seq_input)  # B x L x H
        elif self.args.rnn == 'LSTM':
            output_packed, (hn, cn) = self.rnn(pack_padded_seq_input)  # B x L x H
        else:
            raise NotImplementedError()

        output_padded, output_lengths = pad_packed_sequence(output_packed, batch_first=True)

        scores = self.linear(output_padded)  # B x L x C (C+2)

        if self.args.marryup_type in ['output', 'all']:
            max_len = scores.shape[1]
            scores = scores + re_tags[:, :max_len, :] * self.logits_weights

        flattened_true_label = flatten(label, lengths)

        loss = None
        if self.use_crf:
            mask = get_length_mask(lengths,)

            if train:
                loss = self.crf.neg_log_likelihood_loss(scores, mask, label)

            _, pred_labels = self.crf._viterbi_decode(scores, mask)
            pred_labels = flatten(pred_labels, lengths)

        else:
            flattened_all_scores = flatten(scores, lengths)
            if train:
                loss = self.loss(flattened_all_scores, flattened_true_label)

            _, pred_labels = flattened_all_scores.max(dim=1)

        if self.args.marryup_type == 'kd':
            max_len = scores.shape[1]
            KL_loss = KD_loss(scores, re_tags[:, :max_len, :], self.args)
            loss = self.args.c2_kdpr * loss + (1-self.args.c2_kdpr)*KL_loss

        if self.args.marryup_type == 'pr':
            max_len = scores.shape[1]
            KL_loss = PR_loss(scores, re_tags[:, :max_len, :], self.args)
            pi = max(self.args.c2_kdpr, self.args.c3_pr**self.t)
            loss = pi * loss + (1-pi)*KL_loss

        return loss, pred_labels, flattened_true_label

