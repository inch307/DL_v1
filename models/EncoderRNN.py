"""

Copyright 2017- IBM Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

import sys
import math
import torch.nn as nn

from .baseRNN import BaseRNN

class EncoderRNN(BaseRNN):
    r"""
    Applies a multi-layer RNN to an input sequence.

    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        n_layers (int, optional): number of recurrent layers (default: 1)
        bidirectional (bool, optional): if True, becomes a bidirectional encodr (defulat False)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        variable_lengths (bool, optional): if use variable length RNN (default: False)
        embedding (torch.Tensor, optional): Pre-trained embedding.  The size of the tensor has to match
            the size of the embedding parameter: (vocab_size, hidden_size).  The embedding layer would be initialized
            with the tensor if provided (default: None).
        update_embedding (bool, optional): If the embedding should be updated during training (default: False).

    Inputs: inputs, input_lengths
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is a list of token IDs.
        - **input_lengths** (list of int, optional): list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)

    Outputs: output, hidden
        - **output** (batch, seq_len, hidden_size): tensor containing the encoded features of the input sequence
        - **hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the features in the hidden state `h`

    Examples::

         >>> encoder = EncoderRNN(input_vocab, max_seq_length, hidden_size)
         >>> output, hidden = encoder(input)

    """

    def __init__(self, feature_size, hidden_size,
                 input_dropout_p=0, dropout_p=0,
                 n_layers=1, bidirectional=False, rnn_cell='gru', variable_lengths=False):
        super(EncoderRNN, self).__init__(0, 0, hidden_size,
                input_dropout_p, dropout_p, n_layers, rnn_cell)

        self.variable_lengths = variable_lengths
        

        """
        Copied from https://github.com/SeanNaren/deepspeech.pytorch/blob/master/model.py
        Copyright (c) 2017 Sean Naren
        MIT License
        """
        self.conv = nn.Sequential( # / 4
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        )

        feature_size = math.ceil((feature_size - 11 + 1 + (5*2)) / 2)
        feature_size = math.ceil(feature_size - 11 + 1 + (5*2))
        feature_size *= 32

        self.rnn = self.rnn_cell(feature_size, hidden_size, n_layers,
                                 batch_first=True, bidirectional=bidirectional, dropout=dropout_p)

    def forward(self, input_var, input_lengths=None):
        """
        Applies a multi-layer RNN to an input sequence.

        Args:
            input_var (batch, seq_len): tensor containing the features of the input sequence.
            input_lengths (list of int, optional): A list that contains the lengths of sequences
              in the mini-batch

        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
        """
        
        #(batch, timestep, freq)
        # print("input_var is: " + str(input_var.shape))
        #(batch, 1, timestep, freq)
        input_var = input_var.unsqueeze(1)
        # print("input_var unsqueezed : " + str(input_var.shape))
        x = self.conv(input_var)
        # print("input var conv : " + str(x.shape))

        # 배치, 채널(필터수, 차원), 시간스텝, 주파수별 분석값 => 배치, 시간스텝, 채널(필터수, 차원), 주파수별 분석값
        # BxCxTxD => BxTxCxD
        x = x.transpose(1, 2)
        print("x: " + str(x.shape))
        x = x.contiguous()
        sizes = x.size()
        # 배치, 시간스텝, 채널(필터수, 차원) * 주파수별 분석값
        x = x.view(sizes[0], sizes[1], sizes[2] * sizes[3])

        # input_var is: torch.Size([8, 612, 257])
        # input_var unsqueezed : torch.Size([8, 1, 612, 257])
        # input var conv : torch.Size([8, 32, 153, 129])
        # x: torch.Size([8, 153, 32, 129])
        # input_var is: torch.Size([8, 730, 257])
        # input_var unsqueezed : torch.Size([8, 1, 730, 257])
        # input var conv : torch.Size([8, 32, 183, 129])
        # x: torch.Size([8, 183, 32, 129]) 32*129 = 4128

        # h0 input: (num_layers, 8, 512)
        # hx : (h0, h0)

        if self.training:
            self.rnn.flatten_parameters()

        output, hidden = self.rnn(x)
        print("enc output: " + str(output.shape))
        print("enc hidden: " + str(hidden.shape))
        return output, hidden
