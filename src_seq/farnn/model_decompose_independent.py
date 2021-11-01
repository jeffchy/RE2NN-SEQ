import torch
from torch import nn
from src_seq.utils import flatten, reverse, _maxmul, _matmul, get_length_mask, \
    add_max_val, add_small_constant, add_random_noise
from src_seq.baselines.crf import CRF
import numpy as np
from src_seq.farnn.priority import PriorityLayer
from src_seq.farnn.model_decompose import FARNN_S_D_W


class FARNN_S_D_W_I(FARNN_S_D_W):
    def __init__(self,
                 V=None,
                 S1=None,
                 S2=None,
                 C_output=None,
                 S1_output=None,
                 S2_output=None,
                 wildcard_mat=None,
                 wildcard_output=None,
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
        self.C, self.R_O = C_output.shape
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
                                     S1_output, S2_output,
                                     C_output, wildcard_mat, wildcard_output)

        self.beta = args.beta
        self.beta_vec = nn.Parameter(
            torch.tensor([self.beta] * self.R).float(), requires_grad=bool(self.args.train_beta)
        )
        self.o_idx = o_idx
        self.not_o_idxs = [i for i in range(self.C) if i != self.o_idx]
        self.initialize()

    def init_forward_parameters(self, S1, S2, V, S1_o, S2_o, C_o, W, W_o):
        self.S1 = nn.Parameter(self.pad_additional_states(torch.from_numpy(S1).float()), requires_grad=True) # S x R
        self.S2 = nn.Parameter(self.pad_additional_states(torch.from_numpy(S2).float()), requires_grad=True) # S x R
        self.V_embed = nn.Parameter(torch.from_numpy(V).float(), requires_grad=bool(self.args.train_V_embed)) # V x R
        embed_init_weight = self.embedding.weight.data # V x D
        _pinv = embed_init_weight.pinverse() # D x V
        _V = self.V_embed.data # V x R
        self.embed_r_generalized = nn.Parameter(torch.matmul(_pinv, _V), requires_grad=True) # D x R

        if self.args.use_crf:
            C_o = np.concatenate((C_o, self.get_random((2, self.R_O)).numpy() * self.args.rand_constant),axis=0)
                  # C x R => ( C + 2 ) x R

        self.C_output = nn.Parameter(self.pad_additional_states(torch.from_numpy(C_o).float()),
                                     requires_grad=bool(self.args.train_wildcard)) # C x R_O

        self.S1_output = nn.Parameter(self.pad_additional_states(torch.from_numpy(S1_o).float()),
                                      requires_grad=bool(self.args.train_wildcard)) # S x R_O

        self.S2_output = nn.Parameter(self.pad_additional_states(torch.from_numpy(S2_o).float()),
                                      requires_grad=bool(self.args.train_wildcard)) # S x R_O

        self.wildcard_mat = nn.Parameter(
            self.pad_additional_states(torch.from_numpy(W).float()),
            requires_grad=bool(self.args.train_wildcard_wildcard))

        self.wildcard_output = nn.Parameter(
            self.pad_additional_states(torch.from_numpy(W_o).float()),
            requires_grad=bool(self.args.train_wildcard_wildcard)) if W_o is not None else None


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

            nn.init.xavier_normal_(self.S1_output)
            nn.init.xavier_normal_(self.S2_output)
            nn.init.xavier_normal_(self.C_output)

            nn.init.xavier_normal_(self.embed_r_generalized)
            nn.init.xavier_normal_(self.wildcard_mat)

            nn.init.normal_(self.h0)
            nn.init.normal_(self.hT)

    def get_forward_score(self,
                          hidden_forward,
                          inp,
                          hidden_forward_init,
                          output_tensor_sum,
                          is_forward=True):

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

        temp = torch.einsum('br,sr->bsr', V_vec, self.S1)
        Tr = torch.einsum('sr,bjr->bjs', self.S2, temp)  # S x S batched version is similar
        Tr = Tr + self.wildcard_mat
        Tr = Tr * output_tensor_sum

        if is_forward:
            hidden_forward_next = self.semiring_func(hidden_forward_bar, Tr)  # B x S, B x S x S -> B x S
        else:
            hidden_forward_next = self.semiring_func(hidden_forward_bar, Tr.transpose(1, 2))  # B x S, B x S x S -> B x S

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

    def get_final_score(self, inp, alpha, beta, s1_s2, s1_s2_out):
        V_vec = self.V_embed[inp]  # B x R
        V_vec = self.get_generalized_v_embed_vec(inp, V_vec)
        bss = torch.einsum('ijr,br->bij', s1_s2, V_vec) + self.wildcard_mat
        alpha_beta = torch.einsum('bi,bj->bij', alpha, beta)
        alpha_beta_wild = torch.einsum('bij,bij->bij', alpha_beta, bss)
        br = torch.einsum('bij,rij->br', alpha_beta_wild, s1_s2_out)
        final_score = torch.einsum('br,cr->bc', br, self.C_output)

        return final_score

    def get_output_tensor_sum(self):
        c_output_sum = self.C_output.sum(0) # CxR -> R
        temp = torch.einsum('r,sr->sr', c_output_sum, self.S1_output)
        if self.args.local_loss_func == 'CE1':
            Tr = torch.einsum('sr,jr->js', self.S2_output, temp)
        else:
            Tr = torch.einsum('sr,jr->js', self.S2_output, temp) + self.wildcard_output
        return Tr

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

        output_tensor_sum = self.get_output_tensor_sum() # SxS

        for i in range(L):
            # forward pass
            inp = input[:, i]  # B

            hidden_forward = self.get_forward_score(hidden_forward, inp, hidden_forward_init,
                                                    output_tensor_sum, is_forward=True) # B x S
            forward_scores[:, i, :] = hidden_forward
            # reverse to backward pass
            backward_inp = backward_input[:, i]  # B

            hidden_backward = self.get_forward_score(hidden_backward, backward_inp, hidden_backward_init,
                                                     output_tensor_sum, is_forward=False) # B x S

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
        s1_s2 = torch.einsum('ir,jr->ijr', self.S1, self.S2)
        s1_s2_out = torch.einsum('ir,jr->rij', self.S1_output, self.S2_output)

        for i in range(L):
            inp = input[:, i]  # B
            # accelerated version
            alpha = h0_forward_score[:, i]  # B x S
            beta = reversed_backward_score_x[:, i + 1]  # B x S
            score = self.get_final_score(inp, alpha, beta, s1_s2, s1_s2_out)

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