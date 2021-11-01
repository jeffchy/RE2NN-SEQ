import torch
from torch import nn
from src_seq.utils import flatten, reverse, _maxmul, _matmul, get_length_mask, \
    add_small_constant, add_max_val, add_random_noise
from src_seq.baselines.crf import CRF
import numpy as np
from src_seq.farnn.priority import PriorityLayer
from src_seq.farnn.model_decompose import FARNN_S_D_W
from src_seq.baselines.KD import KD_loss, PR_loss


class FARNN_S_D_W_I_S(FARNN_S_D_W):
    def __init__(self,
                 V=None,
                 S1=None,
                 S2=None,
                 C_output_mat=None,
                 wildcard_mat=None,
                 wildcard_output_vector=None,
                 final_vector=None,
                 start_vector=None,
                 pretrained_word_embed=None,
                 priority_mat=None,
                 args=None,
                 o_idx=0,
                 is_cuda=True):
        """
        Parameters
        ----------
        """
        super(FARNN_S_D_W, self).__init__()

        self.is_cuda = torch.cuda.is_available() if is_cuda else False
        self.additional_states = args.additional_states
        self.args = args
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(pretrained_word_embed).float(), freeze=(not args.train_word_embed)) # V x D

        # V, C, S, S = language_tensor.shape
        self.C, _ = C_output_mat.shape
        self.S, self.R = S1.shape

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

        self.init_forward_parameters(S1, S2, V,
                                     C_output_mat, wildcard_mat, wildcard_output_vector)

        self.beta = args.beta
        self.beta_vec = nn.Parameter(
            torch.tensor([self.beta] * self.R).float(), requires_grad=bool(self.args.train_beta)
        )
        self.o_idx = o_idx
        self.not_o_idxs = [i for i in range(self.C) if i != self.o_idx]

        self.initialize()

    def init_forward_parameters(self, S1, S2, V, C_o, W, W_o):
        self.S1 = nn.Parameter(self.pad_additional_states(torch.from_numpy(S1).float()), requires_grad=True) # S x R
        self.S2 = nn.Parameter(self.pad_additional_states(torch.from_numpy(S2).float()), requires_grad=True) # S x R
        self.V_embed = nn.Parameter(torch.from_numpy(V).float(), requires_grad=bool(self.args.train_V_embed)) # V x R
        embed_init_weight = self.embedding.weight.data # V x D
        _pinv = embed_init_weight.pinverse() # D x V
        _V = self.V_embed.data # V x R
        self.embed_r_generalized = nn.Parameter(torch.matmul(_pinv, _V), requires_grad=True) # D x R

        if self.args.use_crf == 1:
            C_o = np.concatenate((C_o, self.get_random((2, self.S)).numpy() * self.args.rand_constant), axis=0) # C x R => (C+2) x R

        self.C_output_mat = nn.Parameter(self.pad_additional_states(torch.from_numpy(C_o).float()),
                                         requires_grad=bool(self.args.train_c_output))  # C

        self.wildcard_mat = nn.Parameter(
            self.pad_additional_states(torch.from_numpy(W).float()),
            requires_grad=bool(self.args.train_wildcard))

        self.wildcard_output_vector = nn.Parameter(
            self.pad_additional_states(torch.from_numpy(W_o).float()),
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
            nn.init.xavier_normal_(self.V_embed)

            nn.init.xavier_normal_(self.C_output_mat)

            nn.init.xavier_normal_(self.embed_r_generalized)
            nn.init.xavier_normal_(self.wildcard_mat)

            nn.init.normal_(self.h0)
            nn.init.normal_(self.hT)

    def get_forward_score(self, hidden_forward, inp, hidden_forward_init, output_tensor_sum, is_forward=True):

        V_vec = self.V_embed[inp]  # B x R
        V_vec = self.get_generalized_v_embed_vec(inp, V_vec) # B x R

        if self.args.farnn == 0:
            hidden_forward_bar = hidden_forward
        elif self.args.farnn == 1:
            hidden_forward_bar = hidden_forward
            zt = self.gate_activation(torch.matmul(hidden_forward, self.Wss1) + torch.matmul(V_vec, self.Wrs1) + self.bs1)
        elif self.args.farnn == 2:
            zt = self.gate_activation(torch.matmul(hidden_forward, self.Wss1) + torch.matmul(V_vec, self.Wrs1) + self.bs1)
            rt = self.gate_activation(torch.matmul(hidden_forward, self.Wss2) + torch.matmul(V_vec, self.Wrs2) + self.bs2)  # B x S
            hidden_forward_bar = torch.einsum('bs,bs->bs', (1 - rt), hidden_forward_init) + torch.einsum('bs,bs->bs', rt,
                                                                                           hidden_forward)
        else:
            raise NotImplementedError()

        if not is_forward:
            hidden_forward_bar = hidden_forward_bar * output_tensor_sum

        if self.args.train_mode == 'max':
            temp = torch.einsum('br,sr->bsr', V_vec, self.S1)
            Tr = torch.einsum('sr,bjr->bjs', self.S2, temp)  # S x S batched version is similar
            Tr = Tr + self.wildcard_mat
            if is_forward:
                hidden_forward_next = self.semiring_func(hidden_forward_bar, Tr)  # B x S, B x S x S -> B x S
            else:
                hidden_forward_next = self.semiring_func(hidden_forward_bar, Tr.transpose(1, 2))  # B x S, B x S x S -> B x S

        else:
            if is_forward:
                _RR = torch.matmul(hidden_forward_bar, self.S1)  # B x R
                temp = V_vec * _RR  # B x R
                hidden_forward_language = torch.matmul(temp, self.S2.T)  # B x R, R x S  -> B x S
                hidden_forward_wildcard = torch.matmul(hidden_forward_bar, self.wildcard_mat)
            else:
                _RR = torch.matmul(hidden_forward_bar, self.S2)  # B x R
                temp = V_vec * _RR  # B x R
                hidden_forward_language = torch.matmul(temp, self.S1.T)  # B x R, R x S  -> B x S
                hidden_forward_wildcard = torch.matmul(hidden_forward_bar, self.wildcard_mat.T)
            hidden_forward_next = hidden_forward_language + hidden_forward_wildcard

        if is_forward:
            hidden_forward_next = hidden_forward_next * output_tensor_sum

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

    def get_final_score(self, alpha, beta,):
        alpha_beta = alpha * beta # B x S
        final_score = torch.einsum('bs,cs->bc', alpha_beta, self.C_output_mat)
        return final_score

    def forward_local(self, input, label, lengths, train=True, re_tags=None):
        """
        input: Sequence of xs in one sentence, matrix in B x L
        label: Sequence of labels B x L
        lengths: lengths vector in B
        """
        if re_tags is not None:
            B, L, C = re_tags.size()
            if self.args.use_crf:
                re_tags = torch.cat([re_tags, torch.zeros(B, L, 3).to(re_tags.device)], dim=2)
            else:
                re_tags = torch.cat([re_tags, torch.zeros(B, L, 1).to(re_tags.device)], dim=2)

        B, _ = input.size()  # B x L
        L = lengths.max()
        backward_input = reverse(input, lengths)
        hidden_forward = self.h0.unsqueeze(0).repeat(B, 1)  # B x S
        hidden_backward = self.hT.unsqueeze(0).repeat(B, 1)
        hidden_forward_init = self.h0.unsqueeze(0).repeat(B, 1)  # B x S
        hidden_backward_init = self.hT.unsqueeze(0).repeat(B, 1)

        forward_scores = torch.zeros((B, L, self.S + self.additional_states)).cuda() if self.is_cuda else torch.zeros((B, L, self.S + self.additional_states))
        backward_scores = torch.zeros((B, L, self.S + self.additional_states)).cuda() if self.is_cuda else torch.zeros((B, L, self.S + self.additional_states))

        if self.args.local_loss_func == 'CE1':
            output_vector_sum = self.C_output_mat.sum(0)  # C x S -> S
        else:
            output_vector_sum = self.C_output_mat.sum(0) + self.wildcard_output_vector  # C x S -> S

        for i in range(L):
            # forward pass
            inp = input[:, i]  # B

            hidden_forward = self.get_forward_score(hidden_forward, inp, hidden_forward_init,
                                                    output_vector_sum, is_forward=True) # B x S
            forward_scores[:, i, :] = hidden_forward
            # reverse to backward pass
            backward_inp = backward_input[:, i]  # B

            hidden_backward = self.get_forward_score(hidden_backward, backward_inp, hidden_backward_init,
                                                     output_vector_sum, is_forward=False) # B x S

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
            # accelerated version
            alpha = h0_forward_score[:, i + 1]  # B x S
            beta = reversed_backward_score_x[:, i + 1]  # B x S
            score = self.get_final_score(alpha, beta,)

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

            if self.args.marryup_type == 'kd':
                max_len = all_scores.shape[1]
                KL_loss = KD_loss(all_scores, re_tags[:, :max_len, :], self.args)
                loss = self.args.c2_kdpr * loss + (1 - self.args.c2_kdpr) * KL_loss

            if self.args.marryup_type == 'pr':
                max_len = all_scores.shape[1]
                KL_loss = PR_loss(all_scores, re_tags[:, :max_len, :], self.args)
                pi = max(self.args.c2_kdpr, self.args.c3_pr ** self.t)
                loss = pi * loss + (1 - pi) * KL_loss

        flattened_pred_labels = self.decode(all_scores, flattened_all_scores, mask, lengths)

        return loss, flattened_pred_labels, flattened_true_labels


class FARNN_S_SF(FARNN_S_D_W):
    def __init__(self,
                 S1=None,
                 S2=None,
                 C_output_mat=None,
                 wildcard_mat=None,
                 wildcard_output_vector=None,
                 final_vector=None,
                 start_vector=None,
                 priority_mat=None,
                 args=None,
                 o_idx=0,
                 is_cuda=True):
        """
        Parameters
        ----------
        """
        super(FARNN_S_D_W, self).__init__()

        self.is_cuda = torch.cuda.is_available() if is_cuda else False
        self.additional_states = args.additional_states
        self.args = args
        # V, C, S, S = language_tensor.shape
        self.C, _ = C_output_mat.shape
        self.S, self.R = S1.shape

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

        self.init_forward_parameters(S1, S2,
                                     C_output_mat, wildcard_mat, wildcard_output_vector)


        self.o_idx = o_idx
        self.not_o_idxs = [i for i in range(self.C) if i != self.o_idx]

        self.initialize()

    def init_forward_parameters(self, S1, S2, C_o, W, W_o):
        self.S1 = nn.Parameter(self.pad_additional_states(torch.from_numpy(S1).float()), requires_grad=True) # S x R
        self.S2 = nn.Parameter(self.pad_additional_states(torch.from_numpy(S2).float()), requires_grad=True) # S x R

        if self.args.use_crf == 1:
            C_o = np.concatenate((C_o, self.get_random((2, self.S)).numpy() * self.args.rand_constant), axis=0) # C x R => (C+2) x R

        self.C_output_mat = nn.Parameter(self.pad_additional_states(torch.from_numpy(C_o).float()),
                                         requires_grad=bool(self.args.train_c_output))  # C

        self.wildcard_mat = nn.Parameter(
            self.pad_additional_states(torch.from_numpy(W).float()),
            requires_grad=bool(self.args.train_wildcard))

        self.wildcard_output_vector = nn.Parameter(
            self.pad_additional_states(torch.from_numpy(W_o).float()),
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
            nn.init.xavier_normal_(self.C_output_mat)
            nn.init.xavier_normal_(self.wildcard_mat)

            nn.init.normal_(self.h0)
            nn.init.normal_(self.hT)

    def get_forward_score(self, hidden_forward, V_vec, hidden_forward_init, output_tensor_sum, is_forward=True):

        if self.args.farnn == 0:
            hidden_forward_bar = hidden_forward
        elif self.args.farnn == 1:
            hidden_forward_bar = hidden_forward
            zt = self.gate_activation(torch.matmul(hidden_forward, self.Wss1) + torch.matmul(V_vec, self.Wrs1) + self.bs1)
        elif self.args.farnn == 2:
            zt = self.gate_activation(torch.matmul(hidden_forward, self.Wss1) + torch.matmul(V_vec, self.Wrs1) + self.bs1)
            rt = self.gate_activation(torch.matmul(hidden_forward, self.Wss2) + torch.matmul(V_vec, self.Wrs2) + self.bs2)  # B x S
            hidden_forward_bar = torch.einsum('bs,bs->bs', (1 - rt), hidden_forward_init) + torch.einsum('bs,bs->bs', rt,
                                                                                           hidden_forward)
        else:
            raise NotImplementedError()

        if not is_forward:
            hidden_forward_bar = hidden_forward_bar * output_tensor_sum

        if self.args.train_mode == 'max':
            temp = torch.einsum('br,sr->bsr', V_vec, self.S1)
            Tr = torch.einsum('sr,bjr->bjs', self.S2, temp)  # S x S batched version is similar
            Tr = Tr + self.wildcard_mat
            if is_forward:
                hidden_forward_next = self.semiring_func(hidden_forward_bar, Tr)  # B x S, B x S x S -> B x S
            else:
                hidden_forward_next = self.semiring_func(hidden_forward_bar, Tr.transpose(1, 2))  # B x S, B x S x S -> B x S

        else:
            if is_forward:
                _RR = torch.matmul(hidden_forward_bar, self.S1)  # B x R
                temp = V_vec * _RR  # B x R
                hidden_forward_language = torch.matmul(temp, self.S2.T)  # B x R, R x S  -> B x S
                hidden_forward_wildcard = torch.matmul(hidden_forward_bar, self.wildcard_mat)
            else:
                _RR = torch.matmul(hidden_forward_bar, self.S2)  # B x R
                temp = V_vec * _RR  # B x R
                hidden_forward_language = torch.matmul(temp, self.S1.T)  # B x R, R x S  -> B x S
                hidden_forward_wildcard = torch.matmul(hidden_forward_bar, self.wildcard_mat.T)
            hidden_forward_next = hidden_forward_language + hidden_forward_wildcard

        if is_forward:
            hidden_forward_next = hidden_forward_next * output_tensor_sum

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

    def get_final_score(self, alpha, beta,):
        alpha_beta = alpha * beta # B x S
        final_score = torch.einsum('bs,cs->bc', alpha_beta, self.C_output_mat)
        return final_score

    def forward(self, input, label, lengths, train=True, re_tags=None):
        """
        input: embedding setences, matrix in B x L x D
        label: Sequence of labels B x L
        lengths: lengths vector in B
        """
        if re_tags is not None:
            B, L, C = re_tags.size()
            if self.args.use_crf:
                re_tags = torch.cat([re_tags, torch.zeros(B, L, 3).to(re_tags.device)], dim=2)
            else:
                re_tags = torch.cat([re_tags, torch.zeros(B, L, 1).to(re_tags.device)], dim=2)

        B, _, _ = input.size()  # B x L x D
        L = lengths.max()
        backward_input = reverse(input, lengths) # B x L x D
        hidden_forward = self.h0.unsqueeze(0).repeat(B, 1)  # B x S
        hidden_backward = self.hT.unsqueeze(0).repeat(B, 1)
        hidden_forward_init = self.h0.unsqueeze(0).repeat(B, 1)  # B x S
        hidden_backward_init = self.hT.unsqueeze(0).repeat(B, 1)

        forward_scores = torch.zeros((B, L, self.S + self.additional_states)).cuda() if self.is_cuda else torch.zeros((B, L, self.S + self.additional_states))
        backward_scores = torch.zeros((B, L, self.S + self.additional_states)).cuda() if self.is_cuda else torch.zeros((B, L, self.S + self.additional_states))

        if self.args.local_loss_func == 'CE1':
            output_vector_sum = self.C_output_mat.sum(0)  # C x S -> S
        else:
            output_vector_sum = self.C_output_mat.sum(0) + self.wildcard_output_vector  # C x S -> S

        for i in range(L):
            # forward pass
            input_vec = input[:, i, :]  # B x D

            hidden_forward = self.get_forward_score(hidden_forward, input_vec, hidden_forward_init,
                                                    output_vector_sum, is_forward=True) # B x S
            forward_scores[:, i, :] = hidden_forward
            # reverse to backward pass
            backward_input_vec = backward_input[:, i, :]  # B

            hidden_backward = self.get_forward_score(hidden_backward, backward_input_vec, hidden_backward_init,
                                                     output_vector_sum, is_forward=False) # B x S

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
            # accelerated version
            alpha = h0_forward_score[:, i + 1]  # B x S
            beta = reversed_backward_score_x[:, i + 1]  # B x S
            score = self.get_final_score(alpha, beta,)

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

            if self.args.marryup_type == 'kd':
                max_len = all_scores.shape[1]
                KL_loss = KD_loss(all_scores, re_tags[:, :max_len, :], self.args)
                loss = self.args.c2_kdpr * loss + (1 - self.args.c2_kdpr) * KL_loss

            if self.args.marryup_type == 'pr':
                max_len = all_scores.shape[1]
                KL_loss = PR_loss(all_scores, re_tags[:, :max_len, :], self.args)
                pi = max(self.args.c2_kdpr, self.args.c3_pr ** self.t)
                loss = pi * loss + (1 - pi) * KL_loss

        flattened_pred_labels = self.decode(all_scores, flattened_all_scores, mask, lengths)

        return loss, flattened_pred_labels, flattened_true_labels
