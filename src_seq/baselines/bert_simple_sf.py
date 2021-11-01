import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from src_seq.utils import flatten, get_length_mask
from src_seq.baselines.crf import CRF
from src_seq.baselines.KD import KD_loss, PR_loss
from transformers import BertModel
from src_seq.farnn.bert_embeddings import BertEmbedding


class Bert_SF(nn.Module):
    def __init__(self, args, label_size, o_idx):
        """
        Parameters
        ----------
        """
        super(Bert_SF, self).__init__()
        self.args = args
        self.embed = BertEmbedding(args)
        self.is_cuda = torch.cuda.is_available()
        self.use_crf = bool(args.use_crf)
        self.label_size = label_size + 2 if self.use_crf else label_size
        self.crf = CRF(label_size, self.is_cuda)
        self.re_tag_embed = nn.Parameter(torch.randn((self.label_size, args.re_tag_dim)), requires_grad=True)
        self.logits_weights = nn.Parameter(torch.randn(self.label_size), requires_grad=True)  # C,
        self.D = 768
        self.input_dim = self.D + self.args.re_tag_dim if self.args.marryup_type in ['all', 'input'] else self.D
        self.linear = nn.Linear(self.input_dim, self.label_size)
        self.o_idx = o_idx
        self.t = 0

        self.margin = torch.tensor(args.margin).float().cuda() if self.is_cuda else torch.tensor(
            args.margin).float()
        self.epsilon = torch.tensor(1e-5).cuda() if self.is_cuda else torch.tensor(1e-5)

        if self.args.local_loss_func == 'CE':
            self.loss = nn.CrossEntropyLoss()
        elif self.args.local_loss_func == 'ML':
            self.loss = nn.MultiMarginLoss(margin=self.margin)

    def update(self):
        self.t += 1

    def forward(self,
                input,
                bert_input,
                bert_attend_mask,
                bert_valid_mask,
                lengths,
                label,
                train=True,
                re_tags=None):

        inputs = self.embed(bert_input,
                            bert_attend_mask,
                            bert_valid_mask,
                            lengths)  # B x L x D_bert

        B, _, D = inputs.size()
        L = torch.max(lengths)
        re_tags = re_tags[:, :L, :]

        if self.args.use_crf:
            B, L, C = re_tags.size()
            re_tags = torch.cat([re_tags, torch.zeros(B, L, 2).to(re_tags.device)], dim=2)

        if self.args.marryup_type in ['input', 'all']:
            a = (re_tags > 0).sum()
            re_tag_embed_input = torch.einsum('blc,cd->bld', re_tags, self.re_tag_embed) / torch.max(
                                                                        re_tags.sum(-1).unsqueeze(-1), self.epsilon)
            inputs = torch.cat([inputs, re_tag_embed_input], dim=2)

        B, L, D = inputs.size()

        scores = self.linear(inputs)  # B x L x C (C+2)

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

        if train:
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