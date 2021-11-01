import torch
from torch import nn
from src_seq.utils import flatten, reverse, _maxmul, _matmul, \
    bilinear, add_random_noise, add_small_constant, add_max_val
from src_seq.farnn.priority import PriorityLayer


class FARNN_S_O(nn.Module):
    def __init__(self,
                 language_tensor=None,
                 wildcard_tensor=None,
                 wildcard_wildcard_mat=None,
                 final_vector=None,
                 start_vector=None,
                 priority_mat=None,
                 args=None,
                 o_idx=0,
                 is_cuda=False):
        """
        Parameters
        ----------
        """
        super(FARNN_S_O, self).__init__()

        self.is_cuda = torch.cuda.is_available() and is_cuda
        self.args = args
        # V, C, S, S = language_tensor.shape
        C, S, S = wildcard_tensor.shape
        self.S = S
        self.C = C
        self.amp = args.rand_constant
        self.h0 = nn.Parameter(add_random_noise(torch.from_numpy(start_vector).float(), amp=self.amp) , requires_grad=False) # S, trained?
        self.hT = nn.Parameter(add_random_noise(torch.from_numpy(final_vector).float(), amp=self.amp) , requires_grad=False) # S, trained?
        self.language_tensor = nn.Parameter(add_random_noise(torch.from_numpy(language_tensor).float(), amp=self.amp), requires_grad=True) # V x C x  S x S
        self.wildcard_tensor = nn.Parameter(add_random_noise(torch.from_numpy(wildcard_tensor).float(), amp=self.amp), requires_grad=bool(args.train_wildcard)) # C x S x S
        self.wildcard_wildcard_mat = nn.Parameter(
            torch.from_numpy(wildcard_wildcard_mat).float(), requires_grad=bool(args.train_wildcard_wildcard)) # will not use it

        self.priority_layer = PriorityLayer(self.C, priority_mat)
        self.o_idx = o_idx
        self.initialize()

    def initialize(self):
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.relutanh = nn.Sequential(
            nn.ReLU(),
            nn.Tanh()
        )
        self.epsilon = torch.tensor(1e-10).cuda() if self.is_cuda else torch.tensor(1e-10)
        self.margin = torch.tensor(self.args.margin).float().cuda() \
            if self.is_cuda else torch.tensor(self.args.margin).float()
        self.threshold = torch.tensor(self.args.threshold).float().cuda() \
            if self.is_cuda else torch.tensor(self.args.threshold).float()
        self.zero = torch.tensor(0).float().cuda() if self.is_cuda else torch.tensor(0).float()
        self.t = 1
        self.semiring_func = _maxmul if self.args.train_mode == 'max' else _matmul

        if self.args.local_loss_func in ['CE', 'CE1']:
            self.loss = nn.CrossEntropyLoss()
        elif self.args.local_loss_func == 'ML':
            self.loss = nn.MultiMarginLoss(margin=self.margin)
        else:
            raise NotImplementedError()

    def forward_score(self, input, label, lengths, train=True):
        """
                input: Sequence of xs in one sentence, matrix in B x L
                label: Sequence of labels B x L
                lengths: lengths vector in B
                """

        B, L = input.size()  # B x L
        backward_input = reverse(input, lengths)
        hidden_forward = self.h0.unsqueeze(0).repeat(B, 1)  # B x S
        hidden_backward = self.hT.unsqueeze(0).repeat(B, 1)

        forward_scores = torch.zeros((B, L, self.S)).cuda() if self.is_cuda else torch.zeros((B, L, self.S))
        backward_scores = torch.zeros((B, L, self.S)).cuda() if self.is_cuda else torch.zeros((B, L, self.S))

        if self.args.local_loss_func == 'CE1':
            sum_tensor = self.language_tensor.sum(1) + self.wildcard_tensor.sum(0)  # V x S x S
        else:
            sum_tensor = self.language_tensor.sum(1) + self.wildcard_tensor.sum(0) \
                         + self.wildcard_wildcard_mat  # V x S x S

        all_tensor = self.language_tensor + self.wildcard_tensor  # V x C x S x S

        for i in range(L):
            # forward pass
            inp = input[:, i]  # B
            Tr = sum_tensor[inp]
            hidden_forward = self.semiring_func(hidden_forward, Tr)
            hidden_forward = self.relu(hidden_forward)
            forward_scores[:, i, :] = hidden_forward

            # reverse to backward pass
            backward_inp = backward_input[:, i]  # B
            Tr_backward = sum_tensor[backward_inp]
            hidden_backward = self.semiring_func(hidden_backward, Tr_backward.transpose(1, 2))
            hidden_backward = self.relu(hidden_backward)
            backward_scores[:, i, :] = hidden_backward

        # batched version
        # init scores
        all_scores = torch.zeros((B, L, self.C)).cuda() if self.is_cuda else torch.zeros((B, L, self.C))  # B x C x L

        # cat hidden state to forward_score
        h0_forward_score = torch.cat([self.h0.unsqueeze(0).repeat(B, 1, 1), forward_scores, ], dim=1)
        ht_backward_score = torch.cat([self.hT.unsqueeze(0).repeat(B, 1, 1), backward_scores, ], dim=1)

        # add hidden 0
        reversed_backward_score_x = reverse(ht_backward_score, lengths + 1)

        for i in range(L):
            Tr = all_tensor[input[:, i]]  # B x C x S x S
            alpha = h0_forward_score[:, i]  # B x S
            beta = reversed_backward_score_x[:, i + 1]  # B x S
            temp = torch.einsum('bcsj,bs->bcjs', Tr, alpha)
            score = torch.einsum('bcsj,bs->bcsj', temp, beta)
            score = self.relu(score)  # B x C x S x J
            score = score.sum(dim=(2, 3))  # B x C

            if self.args.use_priority:
                score = self.priority_layer(score)

            all_scores[:, i, :] = score

        return all_scores

    def forward_local(self, input, label, lengths, train=True):

        all_scores = self.forward_score(input, label, lengths, train=train)
        # preparing.....
        flattened_true_labels = flatten(label, lengths)
        flattened_all_scores = flatten(all_scores, lengths)

        loss = None

        # training.....
        if train:
            loss = self.loss(flattened_all_scores, flattened_true_labels)

        pred_labels = self.local_decode(flattened_all_scores)

        return loss, pred_labels, flattened_true_labels

    def forward_RE(self, input, label, lengths, train=False):
        all_scores = self.forward_score(input, label, lengths, train=train)

        # predicting, no flat
        if self.args.local_loss_func == 'CE1':
            all_scores = all_scores.clone()
            all_scores[:, :, self.C - 1] = torch.min(all_scores[:, :, self.C - 1], self.threshold)
            max_score, pred_label = all_scores.max(dim=2)
            pred_label[pred_label == self.C - 1] = self.o_idx
        else:
            max_score, pred_label = all_scores.max(dim=2)

        return pred_label, all_scores

    def local_decode(self, all_scores=None):
        """
        input: Sequence of xs in one sentence, matrix in B x L
        label: Sequence of labels B x L
        lengths: lengths vector in B
        scores:
        """

        assert torch.is_tensor(all_scores)

        if self.args.local_loss_func == 'CE1':
            all_scores = all_scores.clone()
            all_scores[:, self.C - 1] = torch.min(all_scores[:, self.C - 1], self.threshold)
            max_score, pred_label = all_scores.max(dim=1)
            pred_label[pred_label == self.C - 1] = self.o_idx
        else:
            max_score, pred_label = all_scores.max(dim=1)

        return pred_label


# FARNN Slot Onehot Independence
class FARNN_S_O_I(FARNN_S_O):
    def __init__(self,
                 language_tensor=None,
                 output_tensor=None,
                 wildcard_mat=None,
                 output_wildcard_mat=None,
                 final_vector=None,
                 start_vector=None,
                 priority_mat=None,
                 args=None,
                 o_idx=0,
                 is_cuda=False):
        """
        Parameters
        ----------
        """
        super(FARNN_S_O, self).__init__()

        self.is_cuda = torch.cuda.is_available() and is_cuda
        self.args = args
        # V, C, S, S = language_tensor.shape
        C, S, S = output_tensor.shape
        self.S = S
        self.C = C
        self.amp = args.rand_constant
        self.h0 = nn.Parameter(add_random_noise(torch.from_numpy(start_vector).float(), amp=self.amp),
                               requires_grad=False)  # S, trained?
        self.hT = nn.Parameter(add_random_noise(torch.from_numpy(final_vector).float(), amp=self.amp),
                               requires_grad=False)  # S, trained?
        self.language_tensor = nn.Parameter(
            add_random_noise(torch.from_numpy(language_tensor).float(), amp=self.amp),
            requires_grad=True)  # V x C x  S x S
        self.wildcard_mat = nn.Parameter(
            add_random_noise(torch.from_numpy(wildcard_mat).float(), amp=self.amp),
            requires_grad=False)  # C x S x S
        self.output_tensor = nn.Parameter(
            torch.from_numpy(output_tensor).float(), requires_grad=False)
        self.output_wildcard_mat = nn.Parameter(
            torch.from_numpy(output_wildcard_mat).float(), requires_grad=False
        ) if output_wildcard_mat is not None else None
        self.priority_layer = PriorityLayer(self.C, priority_mat)

        self.o_idx = o_idx
        self.initialize()

    def get_final_score(self, alpha, beta, Tr):
        alpha_beta = torch.einsum('bi,bj->bij', alpha, beta)
        alpha_beta_tr = torch.einsum('bij,bij->bij', alpha_beta, Tr)
        score = torch.einsum('csj,bsj->bc', self.output_tensor, alpha_beta_tr)
        return score

    def forward_score(self, input, label, lengths, train=True):
        """
                input: Sequence of xs in one sentence, matrix in B x L
                label: Sequence of labels B x L
                lengths: lengths vector in B
                """

        B, L = input.size()  # B x L
        backward_input = reverse(input, lengths)
        hidden_forward = self.h0.unsqueeze(0).repeat(B, 1)  # B x S
        hidden_backward = self.hT.unsqueeze(0).repeat(B, 1)

        forward_scores = torch.zeros((B, L, self.S)).cuda() if self.is_cuda else torch.zeros((B, L, self.S))
        backward_scores = torch.zeros((B, L, self.S)).cuda() if self.is_cuda else torch.zeros((B, L, self.S))

        sum_tensor = self.language_tensor + self.wildcard_mat  # V x S x S
        if self.args.local_loss_func == 'CE1':
            sum_output_tensor = self.output_tensor.sum(0)
        else:
            sum_output_tensor = self.output_tensor.sum(0) + self.output_wildcard_mat

        for i in range(L):
            # forward pass
            inp = input[:, i]  # B
            if self.args.independent == 2:
                Tr = sum_tensor[inp] * sum_output_tensor
            else:
                Tr = sum_tensor[inp]

            hidden_forward = self.semiring_func(hidden_forward, Tr)

            hidden_forward = self.relu(hidden_forward)
            forward_scores[:, i, :] = hidden_forward

            # reverse to backward pass
            backward_inp = backward_input[:, i]  # B
            if self.args.independent == 2:
                Tr_backward = sum_tensor[backward_inp] * sum_output_tensor
            else:
                Tr_backward = sum_tensor[backward_inp]

            hidden_backward = self.semiring_func(hidden_backward, Tr_backward.transpose(1, 2))

            hidden_backward = self.relu(hidden_backward)
            backward_scores[:, i, :] = hidden_backward

        # batched version
        # init scores
        all_scores = torch.zeros((B, L, self.C)).cuda() if self.is_cuda else torch.zeros(
            (B, L, self.C))  # B x C x L

        # cat hidden state to forward_score
        h0_forward_score = torch.cat([self.h0.unsqueeze(0).repeat(B, 1, 1), forward_scores, ], dim=1)
        ht_backward_score = torch.cat([self.hT.unsqueeze(0).repeat(B, 1, 1), backward_scores, ], dim=1)

        # add hidden 0
        reversed_backward_score_x = reverse(ht_backward_score, lengths + 1)

        for i in range(L):
            Tr = sum_tensor[input[:, i]]  # B x S x S
            alpha = h0_forward_score[:, i]  # B x S
            beta = reversed_backward_score_x[:, i + 1]  # B x S
            class_score = self.get_final_score(alpha, beta, Tr)

            all_scores[:, i, :] = class_score

            if self.args.use_priority:
                class_score = self.priority_layer(class_score)

            all_scores[:, i, :] = class_score

        return all_scores


# FARNN Slot Onehot Independence
class FARNN_S_O_I_S(FARNN_S_O):
    def __init__(self, language_tensor=None, output_mat=None, wildcard_mat=None, output_wildcard_vector=None, final_vector=None,
                 start_vector=None, priority_mat=None, args=None, o_idx=0, is_cuda=False):
        """
        Parameters
        ----------
        """
        super(FARNN_S_O, self).__init__()

        self.is_cuda = torch.cuda.is_available() and is_cuda
        self.args = args
        # V, C, S, S = language_tensor.shape
        C, S = output_mat.shape
        self.S = S
        self.C = C
        self.amp = args.rand_constant
        self.h0 = nn.Parameter(add_random_noise(torch.from_numpy(start_vector).float(), amp=self.amp),
                               requires_grad=False)  # S, trained?
        self.hT = nn.Parameter(add_random_noise(torch.from_numpy(final_vector).float(), amp=self.amp),
                               requires_grad=False)  # S, trained?
        self.language_tensor = nn.Parameter(
            add_random_noise(torch.from_numpy(language_tensor).float(), amp=self.amp),
            requires_grad=True)  # V x C x  S x S
        self.wildcard_mat = nn.Parameter(
            add_random_noise(torch.from_numpy(wildcard_mat).float(), amp=self.amp),
            requires_grad=False)  # C x S x S
        self.output_mat = nn.Parameter(
            torch.from_numpy(output_mat).float(), requires_grad=False)
        self.output_wildcard_vector = nn.Parameter(
            torch.from_numpy(output_wildcard_vector).float(), requires_grad=False
        )
        self.priority_layer = PriorityLayer(self.C, priority_mat)

        self.o_idx = o_idx
        self.initialize()

    def get_final_score(self, alpha, beta):
        alpha_beta = alpha * beta # B x S
        score = torch.einsum('cs,bs->bc', self.output_mat, alpha_beta)
        return score

    def forward_score(self, input, label, lengths, train=True):
        """
                input: Sequence of xs in one sentence, matrix in B x L
                label: Sequence of labels B x L
                lengths: lengths vector in B
                """

        B, L = input.size()  # B x L
        backward_input = reverse(input, lengths)
        hidden_forward = self.h0.unsqueeze(0).repeat(B, 1)  # B x S
        hidden_backward = self.hT.unsqueeze(0).repeat(B, 1)

        forward_scores = torch.zeros((B, L, self.S)).cuda() if self.is_cuda else torch.zeros((B, L, self.S))
        backward_scores = torch.zeros((B, L, self.S)).cuda() if self.is_cuda else torch.zeros((B, L, self.S))

        sum_tensor = self.language_tensor + self.wildcard_mat  # V x S x S
        if self.args.local_loss_func == 'CE1':
            sum_output_vector = self.output_mat.sum(0)  # S
        else:
            sum_output_vector = self.output_mat.sum(0) + self.output_wildcard_vector # S

        for i in range(L):
            # forward pass
            inp = input[:, i]  # B
            Tr = sum_tensor[inp]

            hidden_forward = self.semiring_func(hidden_forward, Tr)
            hidden_forward = hidden_forward * sum_output_vector  # B x S, S -> S
            if self.args.update_nonlinear == 'relu':
                hidden_forward = self.relu(hidden_forward)
            elif self.args.update_nonlinear == 'tanh':
                hidden_forward = self.tanh(hidden_forward)
            elif self.args.update_nonlinear == 'relutanh':
                hidden_forward = self.relutanh(hidden_forward)
            else:
                pass
            forward_scores[:, i, :] = hidden_forward

            # reverse to backward pass
            backward_inp = backward_input[:, i]  # B
            Tr_backward = sum_tensor[backward_inp]

            hidden_backward = hidden_backward * sum_output_vector
            hidden_backward = self.semiring_func(hidden_backward, Tr_backward.transpose(1, 2))
            if self.args.update_nonlinear == 'relu':
                hidden_backward = self.relu(hidden_backward)
            elif self.args.update_nonlinear == 'tanh':
                hidden_backward = self.tanh(hidden_backward)
            elif self.args.update_nonlinear == 'relutanh':
                hidden_backward = self.relutanh(hidden_backward)
            else:
                pass
            backward_scores[:, i, :] = hidden_backward

        # batched version
        # init scores
        all_scores = torch.zeros((B, L, self.C)).cuda() if self.is_cuda else torch.zeros(
            (B, L, self.C))  # B x C x L

        # cat hidden state to forward_score
        h0_forward_score = torch.cat([self.h0.unsqueeze(0).repeat(B, 1, 1), forward_scores, ], dim=1)
        ht_backward_score = torch.cat([self.hT.unsqueeze(0).repeat(B, 1, 1), backward_scores, ], dim=1)

        # add hidden 0
        reversed_backward_score_x = reverse(ht_backward_score, lengths + 1)

        for i in range(L):
            # Tr = sum_tensor[input[:, i]]  # B x S x S
            alpha = h0_forward_score[:, i+1]  # B x S
            beta = reversed_backward_score_x[:, i+1]  # B x S
            class_score = self.get_final_score(alpha, beta)

            if self.args.use_priority:
                class_score = self.priority_layer(class_score)

            all_scores[:, i, :] = class_score

        return all_scores