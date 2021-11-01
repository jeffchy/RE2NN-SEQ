from transformers import BertModel
from src_seq.ptm.bert_utils import unflatten_with_lengths
from torch import nn
import torch


class BertEmbedding(nn.Module):
    def __init__(self, args, bert_embed=None):
        super(BertEmbedding, self).__init__()
        self.args = args
        self.bert = BertModel.from_pretrained('bert-base-uncased', local_files_only=True, cache_dir='/p300/huggingface/transformers/').requires_grad_(bool(args.bert_finetune))
        if bert_embed is not None:
            self.static_embed = torch.from_numpy(bert_embed).float()

    def forward(self,
                bert_input,
                bert_attend_mask,
                bert_valid_mask,
                lengths,):

        L = lengths.max()
        outputs = self.bert(input_ids=bert_input, attention_mask=bert_attend_mask,)
        bert_hiddens = outputs[0] # B x L_bert x D_bert
        # select out the valid
        valid_hiddens = bert_hiddens[bert_valid_mask.bool()] # Sum(L) x 768
        padded_bert_hidden = unflatten_with_lengths(valid_hiddens, lengths, L) # B x max_L x 768

        return padded_bert_hidden


class WordEmbedding(nn.Module):
    def __init__(self, args, word_embed):
        super(WordEmbedding, self).__init__()
        self.args = args
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(word_embed).float(), freeze=(not args.train_word_embed)) # V x D
        self.static_embed = torch.from_numpy(word_embed).float()

    def forward(self, inp, lengths):

        # input B x L1
        emb_batch_vec = self.embedding(inp)  # B x L x D

        return emb_batch_vec


class EmbedAggregator(nn.Module):
    def __init__(self, args, V, word_embed):
        super(EmbedAggregator, self).__init__()
        self.args = args
        self.V_embed = nn.Parameter(torch.from_numpy(V).float(), requires_grad=bool(self.args.train_V_embed))  # V x R
        self.V, self.R = self.V_embed.size()

        if bool(self.args.use_bert):
            self.embed = BertEmbedding(args, word_embed)
        else:
            self.embed = WordEmbedding(args, word_embed)

        self.relutanh = nn.Sequential(
            nn.ReLU(),
            nn.Tanh()
        )

        embed_init_weight = self.embed.static_embed # V x D
        _pinv = embed_init_weight.pinverse() # D x V
        _V = self.V_embed.data # V x R
        self.embed_r_generalized = nn.Parameter(torch.matmul(_pinv, _V), requires_grad=True)  # D x R

        self.beta = args.beta
        self.beta_vec = nn.Parameter(
            torch.tensor([self.beta] * self.R).float(), requires_grad=bool(self.args.train_beta)
        )

        if self.args.random:
            nn.init.xavier_normal_(self.V_embed)
            nn.init.xavier_normal_(self.embed_r_generalized)

    def get_generalized_v_embed_vec(self, v_batch_vec, emb_batch_vec):
        emb_batch_vec_generalized = torch.einsum('bld,dr->blr', emb_batch_vec, self.embed_r_generalized)  # B x L x D, D x R -> B x R

        if self.args.additional_nonlinear == 'relu':
            emb_batch_vec_generalized = nn.functional.relu(emb_batch_vec_generalized)
        elif self.args.additional_nonlinear == 'tanh':
            emb_batch_vec_generalized = torch.tanh(emb_batch_vec_generalized)
        elif self.args.additional_nonlinear == 'sigmoid':
            emb_batch_vec_generalized = torch.sigmoid(emb_batch_vec_generalized)
        elif self.args.additional_nonlinear == 'relutanh':
            emb_batch_vec_generalized = self.relutanh(emb_batch_vec_generalized)
        else:
            pass

        generalized_V_batch_vec = v_batch_vec * self.beta_vec + emb_batch_vec_generalized * (1 - self.beta_vec)

        return generalized_V_batch_vec

    def forward(self, inp, lengths):  # not bert
        # input B x L1
        L = lengths.max()
        B, L1 = inp.size()
        if L < L1:
            inp = inp[:, :L] # B x L x D

        emb_batch_vec = self.embed(inp)  # B x L x D
        v_batch_vec = self.V_embed[inp]  # B x L x R
        agg_batch_vec = self.get_generalized_v_embed_vec(v_batch_vec, emb_batch_vec)

        return agg_batch_vec

    def forward_bert(self,
                     inp,
                     bert_input,
                     bert_attend_mask,
                     bert_valid_mask,
                     lengths,): # bert
        # input B x L1
        L = lengths.max()
        B, L1 = inp.size()
        if L < L1:
            inp = inp[:, :L] # B x L x D

        emb_batch_vec = self.embed(bert_input,
                                   bert_attend_mask,
                                   bert_valid_mask,
                                   lengths,)  # B x L x D

        v_batch_vec = self.V_embed[inp]  # B x L x R
        agg_batch_vec = self.get_generalized_v_embed_vec(v_batch_vec, emb_batch_vec)

        return agg_batch_vec
