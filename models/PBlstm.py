import torch
import torch.nn as nn
import logging
import sys
import math

logger = logging.getLogger('root')
FORMAT = "[%(asctime)s %(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logger.setLevel(logging.INFO)
# BLSTM layer for pBLSTM
# Step 1. Reduce time resolution to half
# Step 2. Run through BLSTM
# Note the input should have timestep%2 == 0
class pBLSTMLayer(nn.Module):
    def __init__(self,input_feature_dim,hidden_dim,rnn_unit='LSTM',dropout_rate=0.0):
        super(pBLSTMLayer, self).__init__()
        self.rnn_unit = getattr(nn,rnn_unit.upper())

        # feature dimension will be doubled since time resolution reduction
        self.BLSTM = self.rnn_unit(input_feature_dim*2,hidden_dim,1, bidirectional=True, 
                                   dropout=dropout_rate,batch_first=True)
    
    def forward(self,input_x):
        batch_size = input_x.size(0)
        timestep = input_x.size(1)
        feature_dim = input_x.size(2)
        # print('before input: ' + str(input_x.shape))
        # Reduce time resolution
        input_x = input_x.contiguous().view(batch_size,int(timestep/2),feature_dim*2)
        # print('after input: ' + str(input_x.shape))
        # Bidirectional RNN
        output,hidden = self.BLSTM(input_x)
        return output,hidden

# Listener is a pBLSTM stacking 3 layers to reduce time resolution 8 times
# Input shape should be [# of sample, timestep, features]
class Listener(nn.Module):
    def __init__(self, input_feature_dim, listener_hidden_dim, listener_layer, rnn_unit, use_gpu, dropout_rate=0.0, **kwargs):
        super(Listener, self).__init__()
        # Listener RNN layer
        self.listener_layer = listener_layer
        assert self.listener_layer>=1,'Listener should have at least 1 layer'
        self.rnn_unit = getattr(nn,rnn_unit.upper())
        self.use_gpu = use_gpu
        # if self.use_gpu:
        #     self = self.cuda()
        self.conv = nn.Sequential( # / 4
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        )

        input_feature_dim = math.ceil((input_feature_dim - 11 + 1 + (5*2)) / 2)
        input_feature_dim = math.ceil(input_feature_dim - 11 + 1 + (5*2))
        input_feature_dim *= 32


        self.pLSTM_layer0 = nn.LSTM(input_feature_dim,listener_hidden_dim,1, bidirectional=True, dropout=0,batch_first=True)
        
        # self.pLSTM_layer0 = pBLSTMLayer(input_feature_dim,listener_hidden_dim, rnn_unit=rnn_unit, dropout_rate=dropout_rate)

        for i in range(1,self.listener_layer):
            setattr(self, 'pLSTM_layer'+str(i), pBLSTMLayer(listener_hidden_dim*2,listener_hidden_dim, rnn_unit=rnn_unit, dropout_rate=dropout_rate))

    def forward(self,input_x):
        input_x = input_x.unsqueeze(1)
        # print("input_var unsqueezed : " + str(input_var.shape))
        x = self.conv(input_x)
        # print("input var conv : " + str(x.shape))

        # 배치, 채널(필터수, 차원), 시간스텝, 주파수별 분석값 => 배치, 시간스텝, 채널(필터수, 차원), 주파수별 분석값
        # BxCxTxD => BxTxCxD
        x = x.transpose(1, 2)
        x = x.contiguous()
        sizes = x.size()
        # 배치, 시간스텝, 채널(필터수, 차원) * 주파수별 분석값
        x = x.view(sizes[0], sizes[1], sizes[2] * sizes[3])

        # print('first input: ' + str(input_x.shape))
        self.pLSTM_layer0.flatten_parameters()
        for i in range(1,self.listener_layer):
            getattr(self, 'pLSTM_layer'+str(i)).BLSTM.flatten_parameters()
        output,_  = self.pLSTM_layer0(x)
        hidden = None
        for i in range(1,self.listener_layer):
            output, hidden = getattr(self,'pLSTM_layer'+str(i))(output)
            layer = getattr(self,'pLSTM_layer'+str(i))
        
        # print('pblstm output: ' + str(output.shape))
        return output, hidden

        ############# output, (hidden, cell) = lstm(input, (hidden, cell) )