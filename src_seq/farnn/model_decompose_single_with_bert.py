import torch
from torch import nn
from src_seq.utils import flatten, reverse, _maxmul, _matmul, get_length_mask, \
    add_small_constant, add_max_val, add_random_noise
from src_seq.baselines.crf import CRF
import numpy as np
from src_seq.farnn.priority import PriorityLayer
from src_seq.farnn.model_decompose import FARNN_S_D_W
from src_seq.baselines.KD import KD_loss, PR_loss
from transformers import BertModel
from src_seq.ptm.bert_utils import unflatten_with_lengths
from src_seq.farnn.bert_embeddings import EmbedAggregator
from src_seq.farnn.model_decompose_single import FARNN_S_SF


class FARNN_S_bert(nn.Module):
    def __init__(self,
                 V=None,
                 S1=None,
                 S2=None,
                 C_output_mat=None,
                 wildcard_mat=None,
                 wildcard_output_vector=None,
                 final_vector=None,
                 start_vector=None,
                 static_embed=None,
                 priority_mat=None,
                 args=None,
                 o_idx=0,
                 is_cuda=True):
        """
        Parameters
        ----------
        """
        super(FARNN_S_bert, self).__init__()
        self.embed = EmbedAggregator(args, V, static_embed)
        self.slot_filler = FARNN_S_SF(S1=S1,
                                      S2=S2,
                                      C_output_mat=C_output_mat,
                                      wildcard_mat=wildcard_mat,
                                      wildcard_output_vector=wildcard_output_vector,
                                      final_vector=final_vector,
                                      start_vector=start_vector,
                                      priority_mat=priority_mat,
                                      args=args,
                                      o_idx=o_idx,
                                      is_cuda=is_cuda)

    def forward(self,
                input,
                bert_input,
                bert_attend_mask,
                bert_valid_mask,
                lengths,
                label,
                train=True,
                re_tags=None):

        vecs = self.embed.forward_bert(input,
                                       bert_input,
                                       bert_attend_mask,
                                       bert_valid_mask,
                                       lengths)

        loss, flattened_pred_labels, flattened_true_labels = \
            self.slot_filler(vecs, label, lengths, train=train, re_tags=re_tags)

        return loss, flattened_pred_labels, flattened_true_labels
