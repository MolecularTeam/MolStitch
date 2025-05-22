#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Variable

class MultiGRU(nn.Module):
    """ Implements a three layer GRU cell including an embedding layer
       and an output linear layer back to the size of the vocabulary"""
    def __init__(self, voc_size, cond_dim=None):
        super(MultiGRU, self).__init__()
        self.embedding = nn.Embedding(voc_size, 128)
        self.gru_1 = nn.GRUCell(128, 512)
        self.gru_2 = nn.GRUCell(512, 512)
        self.gru_3 = nn.GRUCell(512, 512)
        if cond_dim:
            self.cond_emb = nn.Linear(cond_dim, 128)
        self.linear = nn.Linear(512, voc_size)

    def forward(self, x, h, cond=None):
        x = self.embedding(x)
        if cond is not None:
            x += self.cond_emb(cond)
        # h_out = torch.zeros(h.size(), device=h.device)
        h_out = Variable(torch.zeros(h.size()))
        x = h_out[0] = self.gru_1(x, h[0])
        x = h_out[1] = self.gru_2(x, h[1])
        x = h_out[2] = self.gru_3(x, h[2])

        x = self.linear(x)
        return x, h_out

    def init_h(self, batch_size):
        # Initial cell state is zero
        return Variable(torch.zeros(3, batch_size, 512))

class RNN():
    """Implements the Prior and Agent RNN. Needs a Vocabulary instance in
    order to determine size of the vocabulary and index of the END token"""
    def __init__(self, voc, cond_dim=None):
        self.rnn = MultiGRU(voc.vocab_size, cond_dim)
        print('need to wait several minutes')
        if torch.cuda.is_available():
            self.rnn.cuda()
        self.voc = voc

    def likelihood(self, target, cond=None):
        """
            Retrieves the likelihood of a given sequence

            Args:
                target: (batch_size * sequence_lenght) A batch of sequences

            Outputs:
                log_probs : (batch_size) Log likelihood for each example*
                entropy: (batch_size) The entropies for the sequences. Not
                                      currently used.
        """
        batch_size, seq_length = target.size()
        start_token = Variable(torch.zeros(batch_size, 1).long())
        start_token[:] = self.voc.vocab['GO']
        x = torch.cat((start_token, target[:, :-1]), 1)
        h = self.rnn.init_h(batch_size)

        log_probs = Variable(torch.zeros(batch_size).float())
        entropy = Variable(torch.zeros(batch_size))
        for step in range(seq_length):
            logits, h = self.rnn(x[:, step], h)
            log_prob = F.log_softmax(logits)
            prob = F.softmax(logits)
            log_probs += NLLLoss(log_prob, target[:, step])
            entropy += -torch.sum((log_prob * prob), 1)
        return log_probs, entropy

    def sample(self, batch_size, cond=None, max_length=140):
        """
            Sample a batch of sequences

            Args:
                batch_size : Number of sequences to sample 
                max_length:  Maximum length of the sequences

            Outputs:
            seqs: (batch_size, seq_length) The sampled sequences.
            log_probs : (batch_size) Log likelihood for each sequence.
            entropy: (batch_size) The entropies for the sequences. Not
                                    currently used.
        """
        start_token = Variable(torch.zeros(batch_size).long())
        start_token[:] = self.voc.vocab['GO']
        h = self.rnn.init_h(batch_size)
        x = start_token

        sequences = []
        log_probs = Variable(torch.zeros(batch_size))
        finished = torch.zeros(batch_size).byte()
        entropy = Variable(torch.zeros(batch_size))
        if torch.cuda.is_available():
            finished = finished.cuda()

        # for step in range(max_length):
        #     logits, h = self.rnn(x, h)
        #     prob = F.softmax(logits)
        #     log_prob = F.log_softmax(logits)
        #     x = torch.multinomial(prob).view(-1)
        #     sequences.append(x.view(-1, 1))
        #     log_probs +=  NLLLoss(log_prob, x)
        #     entropy += -torch.sum((log_prob * prob), 1)

        for step in range(max_length):
            # if step == max_length - 1:
            logits, h = self.rnn(x, h, cond)
            # else:
            #     logits, h = self.rnn(x, h)
            # print('logits shape', logits.shape) ##### [128, 109]
            prob = F.softmax(logits, dim=1)
            log_prob = F.log_softmax(logits, dim=1)
            if torch.isinf(prob).any() or torch.isnan(prob).any():
                prob[torch.isinf(prob)] = 1.0  # inf를 1로 대체
                prob[torch.isnan(prob)] = 0.0  # nan을 0으로 대체

            x = torch.multinomial(prob, num_samples=1).view(-1)  # sampling with prob
            sequences.append(x.view(-1, 1))
            log_probs += NLLLoss(log_prob, x)
            entropy += -torch.sum((log_prob * prob), 1)

            x = Variable(x.data)
            EOS_sampled = (x == self.voc.vocab['EOS']).data
            finished = torch.ge(finished + EOS_sampled, 1)  # if left(input) >= right(other): True
            if torch.prod(finished) == 1: break

        sequences = torch.cat(sequences, 1)
        return sequences.data, log_probs, entropy


    def sample_from_h(self, batch_size, h, cond=None, max_length=140):
        """
            Sample a batch of sequences

            Args:
                batch_size : Number of sequences to sample
                max_length:  Maximum length of the sequences

            Outputs:
            seqs: (batch_size, seq_length) The sampled sequences.
            log_probs : (batch_size) Log likelihood for each sequence.
            entropy: (batch_size) The entropies for the sequences. Not
                                    currently used.
        """
        start_token = Variable(torch.zeros(batch_size).long())
        start_token[:] = self.voc.vocab['GO']
        # h = self.rnn.init_h(batch_size)
        x = start_token

        sequences = []
        log_probs = Variable(torch.zeros(batch_size))
        finished = torch.zeros(batch_size).byte()
        entropy = Variable(torch.zeros(batch_size))
        if torch.cuda.is_available():
            finished = finished.cuda()

        # for step in range(max_length):
        #     logits, h = self.rnn(x, h)
        #     prob = F.softmax(logits)
        #     log_prob = F.log_softmax(logits)
        #     x = torch.multinomial(prob).view(-1)
        #     sequences.append(x.view(-1, 1))
        #     log_probs +=  NLLLoss(log_prob, x)
        #     entropy += -torch.sum((log_prob * prob), 1)

        for step in range(max_length):
            # if step == max_length - 1:
            logits, h = self.rnn(x, h, cond)
            # else:
            #     logits, h = self.rnn(x, h)
            # print('logits shape', logits.shape) ##### [128, 109]
            prob = F.softmax(logits, dim=1)
            log_prob = F.log_softmax(logits, dim=1)
            if torch.isinf(prob).any() or torch.isnan(prob).any():
                prob[torch.isinf(prob)] = 1.0  # inf를 1로 대체
                prob[torch.isnan(prob)] = 0.0  # nan을 0으로 대체

            x = torch.multinomial(prob, num_samples=1).view(-1)  # sampling with prob
            sequences.append(x.view(-1, 1))
            log_probs += NLLLoss(log_prob, x)
            entropy += -torch.sum((log_prob * prob), 1)

            x = Variable(x.data)
            EOS_sampled = (x == self.voc.vocab['EOS']).data
            finished = torch.ge(finished + EOS_sampled, 1)  # if left(input) >= right(other): True
            if torch.prod(finished) == 1: break

        sequences = torch.cat(sequences, 1)
        return sequences.data, log_probs, entropy


    def likelihood_h_out(self, target, cond=None):
        """
            Retrieves the likelihood of a given sequence

            Args:
                target: (batch_size * sequence_lenght) A batch of sequences

            Outputs:
                log_probs : (batch_size) Log likelihood for each example*
                entropy: (batch_size) The entropies for the sequences. Not
                                      currently used.
        """
        batch_size, seq_length = target.size()
        start_token = Variable(torch.zeros(batch_size, 1).long())
        start_token[:] = self.voc.vocab['GO']
        x = torch.cat((start_token, target[:, :-1]), 1)
        h = self.rnn.init_h(batch_size)

        # log_probs = Variable(torch.zeros(batch_size).float())
        # entropy = Variable(torch.zeros(batch_size))
        for step in range(seq_length):
            logits, h = self.rnn(x[:, step], h, cond)
            # log_prob = F.log_softmax(logits)
            # prob = F.softmax(logits)
            # log_probs += NLLLoss(log_prob, target[:, step])
            # entropy += -torch.sum((log_prob * prob), 1)
        return h

    def likelihood_given_h(self, target, h):
        """
            Retrieves the likelihood of a given sequence

            Args:
                target: (batch_size * sequence_lenght) A batch of sequences

            Outputs:
                log_probs : (batch_size) Log likelihood for each example*
                entropy: (batch_size) The entropies for the sequences. Not
                                      currently used.
        """
        batch_size, seq_length = target.size()
        start_token = torch.zeros(batch_size, 1, device=h.device).long()
        start_token[:] = self.voc.vocab['GO']
        x = torch.cat((start_token, target[:, :-1]), 1)
        # h = self.rnn.init_h(batch_size)

        log_probs = torch.zeros(batch_size, device=h.device).float()
        entropy = torch.zeros(batch_size, device=h.device)
        for step in range(seq_length):
            logits, h = self.rnn(x[:, step], h)
            log_prob = F.log_softmax(logits)
            prob = F.softmax(logits)
            log_probs += NLLLoss(log_prob, target[:, step])
            entropy += -torch.sum((log_prob * prob), 1)
        return log_probs, entropy


    def crossover(self, target, cond=None):
        """
            Retrieves the likelihood of a given sequence

            Args:
                target: (batch_size * sequence_lenght) A batch of sequences

            Outputs:
                log_probs : (batch_size) Log likelihood for each example*
                entropy: (batch_size) The entropies for the sequences. Not
                                      currently used.
        """
        batch_size, seq_length = target.size()
        start_token = Variable(torch.zeros(batch_size, 1).long())
        start_token[:] = self.voc.vocab['GO']
        x = torch.cat((start_token, target[:, :-1]), 1)
        h = self.rnn.init_h(batch_size)

        # log_probs = Variable(torch.zeros(batch_size).float())
        # entropy = Variable(torch.zeros(batch_size))
        for step in range(seq_length):
            logits, h = self.rnn(x[:, step], h, cond)
            # log_prob = F.log_softmax(logits)
            # prob = F.softmax(logits)
            # log_probs += NLLLoss(log_prob, target[:, step])
            # entropy += -torch.sum((log_prob * prob), 1)
        return h



def NLLLoss(inputs, targets):
    """
        Custom Negative Log Likelihood loss that returns loss per example,
        rather than for the entire batch.

        Args:
            inputs : (batch_size, num_classes) *Log probabilities of each class*
            targets: (batch_size) *Target class index*

        Outputs:
            loss : (batch_size) *Loss for each example*
    """

    if torch.cuda.is_available():
        target_expanded = torch.zeros(inputs.size()).cuda()
    else:
        target_expanded = torch.zeros(inputs.size())

    target_expanded.scatter_(1, targets.contiguous().view(-1, 1).data, 1.0)
    loss = Variable(target_expanded) * inputs
    loss = torch.sum(loss, 1)
    return loss


class RankModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RankModel, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling over the sequence dimension
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)


    def forward(self, x):
        # x shape: [batch_size, seq_len, feature_dim]
        x = x.permute(0, 2, 1)  # Permute to [batch_size, feature_dim, seq_len]
        x = self.global_avg_pool(x)  # Global average pooling to [batch_size, feature_dim, 1]
        x = x.squeeze(2)  # Remove the last dimension to get [batch_size, feature_dim]
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        # x = torch.tanh(x)
        # x = torch.relu(x)
        return x


class ProxyOracle(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ProxyOracle, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling over the sequence dimension
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: [batch_size, seq_len, feature_dim]
        x = x.permute(0, 2, 1)  # Permute to [batch_size, feature_dim, seq_len]
        x = self.global_avg_pool(x)  # Global average pooling to [batch_size, feature_dim, 1]
        x = x.squeeze(2)  # Remove the last dimension to get [batch_size, feature_dim]
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        # x = torch.tanh(x)
        x = torch.relu(x)
        return x



