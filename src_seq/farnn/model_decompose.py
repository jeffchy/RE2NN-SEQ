import torch
from torch import nn
from src_seq.utils import flatten, reverse, _maxmul, _matmul, get_length_mask, bilinear, \
    add_random_noise, add_small_constant, add_max_val
from src_seq.baselines.crf import CRF
import numpy as np
from src_seq.farnn.priority import PriorityLayer


class FARNN_S_D_W(nn.Module):
    def __init__(self,
                 V=None,
                 C=None,
                 S1=None,
                 S2=None,
                 C_wildcard=None,
                 S1_wildcard=None,
                 S2_wildcard=None,
                 wildcard_wildcard=None,
                 final_vector=None,
                 start_vector=None,
                 pretrained_word_embed=None,
                 priority_mat=None,
                 args=None,
                 o_idx=0):
        """
        Parameters
        ----------
        """
        super(FARNN_S_D_W, self).__init__()

        self.is_cuda = torch.cuda.is_available()
        self.additional_states = args.additional_states
        self.args = args
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(pretrained_word_embed).float(), freeze=(not args.train_word_embed)) # V x D

        # V, C, S, S = language_tensor.shape
        self.C, self.R_W = C_wildcard.shape
        self.S, _ = S1_wildcard.shape
        _, self.R = C.shape

        self.t = 1
        self.use_crf = bool(args.use_crf)

        if self.use_crf:
            self.crf = CRF(self.C, self.is_cuda)
            self.C += 2

        self.priority_layer = PriorityLayer(self.C, priority_mat)

        self.random = bool(args.random)
        self.h0 = nn.Parameter(
            self.pad_additional_states(torch.from_numpy(start_vector).float()), requires_grad=bool(args.train_h0))  # S hidden state dim should be equal to the state dim
        self.hT = nn.Parameter(
            self.pad_additional_states(torch.from_numpy(final_vector).float()), requires_grad=bool(args.train_hT)) # S, trained?

        self.init_forward_parameters(S1, S2, C, V,
                                     S1_wildcard, S2_wildcard,
                                     C_wildcard, wildcard_wildcard)

        self.beta = args.beta
        self.beta_vec = nn.Parameter(
            torch.tensor([self.beta] * self.R).float(), requires_grad=bool(self.args.train_beta)
        )
        self.o_idx = o_idx
        self.not_o_idxs = [i for i in range(self.C) if i != self.o_idx]
        self.initialize()

    def initialize(self):
        self.additional_nonlinear = self.args.additional_nonlinear

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.relutanh = nn.Sequential(
            nn.ReLU(),
            nn.Tanh()
        )
        self.semiring_func = _matmul if self.args.train_mode == 'sum' else _maxmul
        self.nllloss = nn.NLLLoss()
        self.celoss = nn.CrossEntropyLoss()
        self.epsilon = torch.tensor(1e-16).cuda() if self.is_cuda else torch.tensor(1e-16)
        self.margin = torch.tensor(self.args.margin).float().cuda() if self.is_cuda else torch.tensor(
            self.args.margin).float()
        self.threshold = torch.tensor(self.args.threshold).float().cuda() if self.is_cuda else torch.tensor(
            self.args.threshold).float()
        self.zero = torch.tensor(0).float().cuda() if self.is_cuda else torch.tensor(0).float()

        self.gate_activation = self.Sigmoidal(self.args.sigmoid_exponent)

        if self.args.local_loss_func in ['CE', 'CE1']:
            self.loss = self.celoss
        elif self.args.local_loss_func == 'ML':
            self.loss = nn.MultiMarginLoss(margin=self.margin)
        else:
            raise NotImplementedError()

    def Sigmoidal(self, exponent):
        def func(x):
            assert exponent > 0
            input = x * exponent
            return nn.functional.sigmoid(input)
        return func

    def init_forward_parameters(self, S1, S2, C, V, S1_w, S2_w, C_w, W):
        self.S1 = nn.Parameter(self.pad_additional_states(torch.from_numpy(S1).float()), requires_grad=True) # S x R
        self.S2 = nn.Parameter(self.pad_additional_states(torch.from_numpy(S2).float()), requires_grad=True) # S x R
        self.V_embed = nn.Parameter(torch.from_numpy(V).float(), requires_grad=bool(self.args.train_V_embed)) # V x R
        embed_init_weight = self.embedding.weight.data # V x D
        _pinv = embed_init_weight.pinverse() # D x V
        _V = self.V_embed.data # V x R
        self.embed_r_generalized = nn.Parameter(torch.matmul(_pinv, _V), requires_grad=True) # D x R


        if self.args.use_crf == 1:
            C = np.concatenate((C, self.get_random((2, self.R)).numpy() * self.args.rand_constant), axis=0) # C x R => (C+2) x R
            C_w = np.concatenate((C_w, self.get_random((2, self.R_W)).numpy() * self.args.rand_constant), axis=0)


        self.C_wildcard = nn.Parameter(self.pad_additional_states(torch.from_numpy(C_w).float()),
                                        requires_grad=bool(self.args.train_wildcard))  # C x R

        self.C_embed = nn.Parameter(torch.from_numpy(C).float(), requires_grad=True)  # (C+2) x R

        self.S1_wildcard = nn.Parameter(self.pad_additional_states(torch.from_numpy(S1_w).float()),
                                                requires_grad=bool(self.args.train_wildcard)) # S x R

        self.S2_wildcard = nn.Parameter(self.pad_additional_states(torch.from_numpy(S2_w).float()),
                                        requires_grad=bool(self.args.train_wildcard)) # S x R

        self.wildcard_wildcard = nn.Parameter(
            self.pad_additional_states(torch.from_numpy(W).float()),
            requires_grad=bool(self.args.train_wildcard_wildcard))


        # parameters for GRU (if farnn==0, normal rnn)
        if self.args.farnn == 1:
            self.Wss1 = nn.Parameter((torch.randn((self.S + self.additional_states, self.S + self.additional_states))).float(), requires_grad=True)
            self.Wrs1 = nn.Parameter((torch.randn((self.R, self.S + self.additional_states))).float(), requires_grad=True)
            self.bs1 = nn.Parameter((torch.ones((1, self.S + self.additional_states)).float() * self.args.bias_init ), requires_grad=True)
            if self.args.xavier:
                nn.init.xavier_normal_(self.Wss1)
                nn.init.xavier_normal_(self.Wrs1)
            if self.args.xavier:
                nn.init.xavier_normal_(self.bs1)

        if self.args.farnn == 2:
            self.Wss1 = nn.Parameter((torch.randn(
                (self.S + self.additional_states, self.S + self.additional_states))).float(),
                                     requires_grad=True)
            self.Wrs1 = nn.Parameter((torch.randn(
                (self.R, self.S + self.additional_states))).float(),
                                     requires_grad=True)
            self.bs1 = nn.Parameter((torch.ones((1, self.S + self.additional_states)).float() * self.args.bias_init),
                                    requires_grad=True)
            self.Wss2 = nn.Parameter((torch.randn((
                self.S + self.additional_states, self.S + self.additional_states))).float(),
                                     requires_grad=True)
            self.Wrs2 = nn.Parameter((torch.randn((self.R, self.S + self.additional_states))).float(),
                                     requires_grad=True)
            self.bs2 = nn.Parameter((torch.ones((1, self.S + self.additional_states)).float() * self.args.bias_init),
                                    requires_grad=True)
            if self.args.xavier:
                nn.init.xavier_normal_(self.Wss1)
                nn.init.xavier_normal_(self.Wrs1)
                nn.init.xavier_normal_(self.Wss2)
                nn.init.xavier_normal_(self.Wrs2)

        if self.random:
            nn.init.xavier_normal_(self.S1)
            nn.init.xavier_normal_(self.S2)
            nn.init.xavier_normal_(self.C_embed)
            nn.init.xavier_normal_(self.V_embed)

            nn.init.xavier_normal_(self.S1_wildcard)
            nn.init.xavier_normal_(self.S2_wildcard)
            nn.init.xavier_normal_(self.C_wildcard)

            nn.init.xavier_normal_(self.embed_r_generalized)
            nn.init.xavier_normal_(self.wildcard_wildcard)

            nn.init.normal_(self.h0)
            nn.init.normal_(self.hT)

    def get_prior_tensor(self):
        new_tensor = self.wildcard_wildcard.repeat(self.C, 1, 1)
        prior = torch.ones(self.C).cuda() if self.is_cuda else torch.ones(self.C)
        prior[self.o_idx] = 1
        prior = prior / prior.sum()
        return prior.reshape(self.C, 1, 1) * new_tensor

    def get_random(self, sizes):
        if self.args.random_pad_func == 'uniform':
            return torch.rand(sizes)
        elif self.args.random_pad_func == 'normal':
            return torch.randn(sizes)
        else:
            a = torch.randn(sizes)
            nn.init.xavier_normal_(a)
            return a

    def pad_additional_states(self, obj):
        origin_shape = obj.shape
        padded_shape = tuple([i if i != self.S else i + self.additional_states for i in origin_shape])
        if len(origin_shape) == 1:
            small_random = torch.zeros(padded_shape)
        else:
            small_random = self.get_random(padded_shape) * self.args.rand_constant

        if len(origin_shape) == 1:
            small_random[:origin_shape[0]] = obj
        elif len(origin_shape) == 2:
            small_random[:origin_shape[0], :origin_shape[1],] = obj
        elif len(origin_shape) == 3:
            small_random[:origin_shape[0], :origin_shape[1], :origin_shape[2]] = obj
        elif len(origin_shape) == 4:
            small_random[:origin_shape[0], :origin_shape[1], :origin_shape[2], :origin_shape[3]] = obj
        else:
            raise NotImplementedError()

        return small_random

    def get_generalized_v_embed_vec(self, inp, v_vec):

        emb_batch_vec = self.embedding(inp)  # B x D

        emb_batch_vec_generalized = torch.matmul(emb_batch_vec, self.embed_r_generalized)  # B x D, D x R -> B x R

        if self.additional_nonlinear == 'relu':
            emb_batch_vec_generalized = nn.functional.relu(emb_batch_vec_generalized)
        elif self.additional_nonlinear == 'tanh':
            emb_batch_vec_generalized = torch.tanh(emb_batch_vec_generalized)
        elif self.additional_nonlinear == 'sigmoid':
            emb_batch_vec_generalized = torch.sigmoid(emb_batch_vec_generalized)
        elif self.additional_nonlinear == 'relutanh':
            emb_batch_vec_generalized = self.relutanh(emb_batch_vec_generalized)
        else:
            pass

        generalized_V_batch_vec = v_vec * self.beta_vec + emb_batch_vec_generalized * (1 - self.beta_vec)

        return generalized_V_batch_vec

    def get_forward_score(self,
                          hidden_forward,
                          inp,
                          C_vec_sum,
                          wildcard_tensor_origin_sum,
                          hidden_forward_init,
                          is_forward=True):

        V_vec = self.V_embed[inp]  # B x R
        V_vec = self.get_generalized_v_embed_vec(inp, V_vec)
        _R = V_vec * C_vec_sum  # R

        if self.args.farnn == 0:
            hidden_forward_bar = hidden_forward
        elif self.args.farnn == 1:
            hidden_forward_bar = hidden_forward
            zt = self.gate_activation(torch.matmul(hidden_forward, self.Wss1) + torch.matmul(_R, self.Wrs1) + self.bs1)
        elif self.args.farnn == 2:
            zt = self.gate_activation(torch.matmul(hidden_forward, self.Wss1) + torch.matmul(_R, self.Wrs1) + self.bs1)
            rt = self.gate_activation(torch.matmul(hidden_forward, self.Wss2) + torch.matmul(_R, self.Wrs2) + self.bs2)  # B x S
            hidden_forward_bar = torch.einsum('bs,bs->bs', (1 - rt), hidden_forward_init) + torch.einsum('bs,bs->bs', rt,
                                                                                           hidden_forward)
        else:
            raise NotImplementedError()

        # Tr = sum_tensor[inp] # B x S x S
        if self.args.train_mode == 'max':
            temp = torch.einsum('br,sr->bsr', _R, self.S1)
            Tr = torch.einsum('sr,bjr->bjs', self.S2, temp)  # S x S batched version is similar
            Tr = Tr + wildcard_tensor_origin_sum
            if is_forward:
                hidden_forward_next = _maxmul(hidden_forward_bar, Tr)  # B x R, B x R -> B x R
            else:
                hidden_forward_next = _maxmul(hidden_forward_bar, Tr.transpose(1, 2))  # B x R, B x R -> B x R

        else:
            if is_forward:
                _RR = torch.matmul(hidden_forward_bar, self.S1)  # B x R
                temp = _R * _RR  # B x R
                hidden_forward_language = torch.matmul(temp, self.S2.T)  # B x R, R x S  -> B x S
                hidden_forward_wildcard = torch.matmul(hidden_forward_bar, wildcard_tensor_origin_sum)
                hidden_forward_next = hidden_forward_language + hidden_forward_wildcard
            else:
                _RR = torch.matmul(hidden_forward_bar, self.S2)  # B x R
                temp = _R * _RR  # B x R
                hidden_forward_language = torch.matmul(temp, self.S1.T)  # B x R, R x S  -> B x S
                hidden_forward_wildcard = torch.matmul(hidden_forward_bar, wildcard_tensor_origin_sum.T)
                hidden_forward_next = hidden_forward_language + hidden_forward_wildcard

        if self.args.update_nonlinear == 'relu':
            hidden_forward_next = self.relu(hidden_forward_next)
        elif self.args.update_nonlinear == 'tanh':
            hidden_forward_next = self.tanh(hidden_forward_next)
        elif self.args.update_nonlinear == 'relutanh':
            hidden_forward_next = self.relutanh(hidden_forward_next)
        else:
            pass

        if self.args.farnn == 0:
            hidden_forward = hidden_forward_next
        elif self.args.farnn in [1, 2]:
            hidden_forward = torch.einsum('bs,bs->bs', (1 - zt), hidden_forward) + torch.einsum('bs,bs->bs', zt, hidden_forward_next)
        else:
            raise NotImplementedError()

        return hidden_forward

    def get_final_score(self, inp, alpha, beta):

        V_vec = self.V_embed[inp]  # B x R
        V_vec = self.get_generalized_v_embed_vec(inp, V_vec)
        temp = torch.einsum('br,cr->bcr', V_vec, self.C_embed)  # B x C x R
        alpha_S1 = torch.matmul(alpha, self.S1)  # B x R
        beta_S2 = torch.matmul(beta, self.S2)  # B x R
        alpha_beta = alpha_S1 * beta_S2  # B x R
        score = torch.einsum('bcr,br->bc', temp, alpha_beta)
        br_alpha = torch.matmul(alpha, self.S1_wildcard)
        br_beta = torch.matmul(beta, self.S2_wildcard)
        br_alpha_beta = br_alpha * br_beta
        score_wildcard = torch.einsum('br,cr->bc', br_alpha_beta, self.C_wildcard)
        score = score + score_wildcard

        return score

    def get_wildcard_tensor_origin_sum_forward(self):
        C_wildcard_sum = self.C_wildcard.sum(0) # R
        temp = torch.einsum('r,sr->rs', C_wildcard_sum, self.S1_wildcard)
        res = torch.einsum('sr,rj->js', self.S2_wildcard, temp)  # S x S batched version is similar

        return res + self.wildcard_wildcard

    def get_wildcard_tensor_forward(self):
        temp = torch.einsum('cr,sr->crs', self.C_wildcard, self.S1_wildcard)
        res = torch.einsum('sr,crj->cjs', self.S2_wildcard, temp)  # S x S batched version is similar
        return res

    def decode(self, all_scores, flattened_all_scores, mask, lengths):
        B, L, C = all_scores.shape
        # for analyze.....
        # print()
        # print('MEAN: {}'.format(flattened_all_scores.mean(0)))
        # print('STD: {}'.format(flattened_all_scores.std(0)))
        # print('MAX: {}'.format(flattened_all_scores.max(0)[0]))
        # print('MIN: {}'.format(flattened_all_scores.min(0)[0]))

        # decoding.....
        if self.use_crf:

            if self.args.local_loss_func == 'CE1':
                all_scores = all_scores.clone() # N x C
                all_scores[:, :, self.C - 3] = torch.min(all_scores[:, :, self.C - 3], self.threshold)
                scores, pred_label = self.crf._viterbi_decode(all_scores, mask)
                pred_label = flatten(pred_label, lengths)
                pred_label[pred_label == self.C - 3] = self.o_idx  # minus three because we add start/end tag
            else:
                scores, pred_label = self.crf._viterbi_decode(all_scores, mask)
                pred_label = flatten(pred_label, lengths)

        else:

            if self.args.local_loss_func == 'CE1':
                all_scores = flattened_all_scores.clone() # N x C
                all_scores[:, self.C - 1] = torch.min(all_scores[:, self.C - 1], self.threshold)
                max_score, pred_label = all_scores.max(dim=1)
                pred_label[pred_label == self.C - 1] = self.o_idx
            else:
                max_score, pred_label = flattened_all_scores.max(dim=1)

        return pred_label

    def forward_local(self, input, label, lengths, train=True):
        """
        input: Sequence of xs in one sentence, matrix in B x L
        label: Sequence of labels B x L
        lengths: lengths vector in B
        """

        B, _ = input.size()  # B x L
        L = lengths.max()
        backward_input = reverse(input, lengths)
        hidden_forward = self.h0.unsqueeze(0).repeat(B, 1)  # B x S
        hidden_backward = self.hT.unsqueeze(0).repeat(B, 1)
        hidden_forward_init = self.h0.unsqueeze(0).repeat(B, 1)  # B x S
        hidden_backward_init = self.hT.unsqueeze(0).repeat(B, 1)

        forward_scores = torch.zeros((B, L, self.S + self.additional_states)).cuda() if self.is_cuda else torch.zeros((B, L, self.S + self.additional_states))
        backward_scores = torch.zeros((B, L, self.S + self.additional_states)).cuda() if self.is_cuda else torch.zeros((B, L, self.S + self.additional_states))

        # wildcard_tensor_origin = self.wildcard_tensor
        wildcard_tensor_origin_sum = self.get_wildcard_tensor_origin_sum_forward()

        C_vec_sum = self.C_embed.sum(0)  # R

        for i in range(L):
            # forward pass
            inp = input[:, i]  # B

            hidden_forward = self.get_forward_score(hidden_forward, inp, C_vec_sum, wildcard_tensor_origin_sum,
                                                    hidden_forward_init, is_forward=True)
            forward_scores[:, i, :] = hidden_forward
            # reverse to backward pass
            backward_inp = backward_input[:, i]  # B

            hidden_backward = self.get_forward_score(hidden_backward, backward_inp, C_vec_sum, wildcard_tensor_origin_sum,
                                                     hidden_backward_init, is_forward=False)

            backward_scores[:, i, :] = hidden_backward

        # batched version
        # init scores
        all_scores = torch.zeros((B, L, self.C)).cuda() if self.is_cuda else torch.zeros(
            (B, L, self.C))  # B x L x C

        # cat hidden state to forward_score
        h0_forward_score = torch.cat([self.h0.unsqueeze(0).repeat(B, 1, 1), forward_scores, ], dim=1)
        ht_backward_score = torch.cat([self.hT.unsqueeze(0).repeat(B, 1, 1), backward_scores, ], dim=1)

        # add hidden 0
        reversed_backward_score_x = reverse(ht_backward_score, lengths + 1)

        for i in range(L):
            inp = input[:, i]
            # accelerated version
            alpha = h0_forward_score[:, i]  # B x S
            beta = reversed_backward_score_x[:, i + 1]  # B x S

            score = self.get_final_score(inp, alpha, beta)

            all_scores[:, i, :] = score

        if self.args.use_priority:
            all_scores = self.priority_layer(all_scores)

        loss = None
        mask = None

        if self.use_crf:
            mask = get_length_mask(lengths, )

        # preparing.....
        flattened_true_labels = flatten(label, lengths)
        flattened_all_scores = flatten(all_scores, lengths)

        # training.....
        if train:
            if self.use_crf:
                loss = self.crf.neg_log_likelihood_loss(all_scores, mask, label)
            else:
                loss = self.loss(flattened_all_scores, flattened_true_labels)

        # decoding
        flattened_pred_labels = self.decode(all_scores, flattened_all_scores, mask, lengths)

        return loss, flattened_pred_labels, flattened_true_labels



