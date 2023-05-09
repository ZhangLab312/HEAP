import collections

import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import itertools
import read_data
from options import options
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
import copy, math
from torch.autograd import Variable

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path = "./sequence"

THRESHOLD = .1
INITIALIZATION = .15


def upsample(x, out_size):
    return F.interpolate(x, size=out_size, mode='linear', align_corners=False)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        self.pe = torch.zeros(max_len, d_model)  # [max_len,d_model]
        position = torch.arange(0., max_len).unsqueeze(1)  # [max_len,1]
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))  # [1,d_model/2]
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)  # [1,max_len,d_model]

    def forward(self, x):  # x = [1,wordnum,d_model]

        x = x + Variable(self.pe[:, :x.size(1)].to(device),
                         requires_grad=False)
        return self.dropout(x)


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, N, size, dropout, d_model, d_ff, h, pos_embed):
        super(Encoder, self).__init__()
        # embedding
        self.pos_embed = pos_embed
        # layernorm
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)
        self.layernorm4 = nn.LayerNorm(d_model)
        # multihead_attention
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear11 = Mlp_Freeze(d_model, d_model)
        self.linear12 = Mlp_Freeze(d_model, d_model)
        self.linear13 = Mlp_Freeze(d_model, d_model)
        self.linear14 = Mlp_Freeze(d_model, d_model)

        self.linear21 = Mlp_Freeze(d_model, d_model)
        self.linear22 = Mlp_Freeze(d_model, d_model)
        self.linear23 = Mlp_Freeze(d_model, d_model)
        self.linear24 = Mlp_Freeze(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

        self.N = N

        # self.feed_forward = feed_forward
        self.w_11 = Mlp_Freeze(d_model, d_ff)  # encoder 1
        self.w_21 = Mlp_Freeze(d_ff, d_model)
        # self.dropout = nn.Dropout(dropout)

        self.w_12 = Mlp_Freeze(d_model, d_ff)  # encoder 2
        self.w_22 = Mlp_Freeze(d_ff, d_model)

        # self.sublayer = clones(SublayerConnection(size, dropout), 2)
        # self.size = size

    def forward(self, x, mask=None):
        "Pass the input (and mask) through each layer in turn."
        x = self.pos_embed(x)
        temp = x
        query, key, value = x, x, x
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query = self.linear11(query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        key = self.linear12(key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value = self.linear13(value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)

        x = attention(query, key, value, mask=mask,
                      dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        x = self.linear14(x)
        x = x + temp
        x = self.layernorm1(x)
        temp = x
        x = self.w_21(self.w_11(x))
        x = x + temp
        x = self.layernorm2(x)

        # encoder 2
        temp = x
        query, key, value = x, x, x
        if mask is not None:
            # Same mask applied to module h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query = self.linear21(query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        key = self.linear22(key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value = self.linear23(value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)

        x = attention(query, key, value, mask=mask,
                      dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        x = self.linear24(x)
        x = x + temp
        x = self.layernorm3(x)
        temp = x
        x = self.w_22(self.w_12(x))
        x = x + temp
        x = self.layernorm4(x)
        return x


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value)  # !!


def make_model(NTrans, d_model, d_ff, h, dropout=0.1):
    c = copy.deepcopy
    position = PositionalEncoding(1, dropout)
    model = Encoder(NTrans, d_model, dropout, d_model, d_ff, h, c(position))

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class HEAP_DNA(nn.Module):
    def __init__(self, motiflen=9):
        super(HEAP_DNA, self).__init__()
        # encoding
        self.trans1 = make_model(1, 64, 128, 8)
        self.trans2 = make_model(1, 64, 128, 8)
        self.conv = freeze_conv(in_channels=4, out_channels=64, kernel_size=motiflen, padding='same')
        self.conv1 = freeze_conv(in_channels=64, out_channels=64, kernel_size=motiflen, padding='same')
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = freeze_conv(in_channels=64, out_channels=64, kernel_size=motiflen, padding='same')
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = freeze_conv(in_channels=64, out_channels=64, kernel_size=motiflen, padding='same')
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # decoding
        self.blend_conv3 = freeze_conv(in_channels=64, out_channels=64, kernel_size=motiflen, padding='same')
        self.blend_conv2 = freeze_conv(in_channels=64, out_channels=64, kernel_size=motiflen, padding='same')
        self.blend_conv1 = freeze_conv(64, 4, kernel_size=motiflen, padding='same')
        self.out = freeze_conv(4, 1, kernel_size=motiflen, padding='same')
        self.batchnorm64_3 = nn.BatchNorm1d(64)
        self.batchnorm64_2 = nn.BatchNorm1d(64)
        self.batchnorm4 = nn.BatchNorm1d(64)
        # general functions
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()
        self.flatten = nn.Flatten(start_dim=1)

    def _init_weights(self):
        """Initialize the new built layers"""
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    def getIndicators(self):
        indicators = []
        for i in self.named_parameters():
            if 'indicator' in i[0]:
                indicators.append(i[1])
        return indicators

    def forward(self, x):

        x = torch.transpose(x, 1, 2)
        b, _, _ = x.size()
        # encoding
        out1 = self.conv(x)
        res1 = out1
        out1 = self.conv1(out1)
        out1 = self.relu(out1)
        out1 = self.pool1(out1)
        out1 = self.dropout(out1)
        res2 = out1
        out1 = self.conv2(out1)
        out1 = self.relu(out1)
        out1 = self.pool2(out1)
        out1 = self.dropout(out1)
        res3 = out1
        out1 = self.conv3(out1)
        out1 = self.relu(out1)
        out1 = self.pool3(out1)
        out1 = self.dropout(out1)
        out1 = out1.permute(0, 2, 1)
        out1_1 = self.trans1(out1)
        out1_2 = self.trans2(torch.flip(out1, [1]))
        out1 = out1_1 + out1_2
        out1 = self.dropout(out1)
        res4 = out1.permute(0, 2, 1)

        up3 = upsample(res4, res3.size()[-1])
        up3 = up3 + res3
        up3 = self.batchnorm64_3(up3)
        up3 = self.relu(up3)
        up3 = self.blend_conv3(up3)
        up2 = upsample(up3, res2.size()[-1])
        up2 = up2 + res2

        up2 = self.batchnorm64_2(up2)
        up2 = self.relu(up2)
        up2 = self.blend_conv2(up2)

        up1 = upsample(up2, res1.size()[-1])
        up1 = up1 + res1
        up1 = self.batchnorm4(up1)
        up1 = self.relu(up1)
        out_dense = self.blend_conv1(up1)
        out_dense = self.out(out_dense)
        out_dense = self.flatten(out_dense)
        # out_dense = self.linear(out_dense)
        # out = self.fc(out_dense)
        # out = self.softmax(out)
        return out_dense


class Mlp_Freeze(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, out_features, act_layer=nn.ReLU, drop=0.2):
        super().__init__()

        # out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, out_features)
        # self.fc1.weight.requires_grad = False
        # self.fc1.weight.grad = None
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        # self.register_parameter(name = 'weight_delta', param = torch.nn.Parameter(torch.zeros(out_features,in_features)))
        # self.register_parameter(name = 'indicator', param = torch.nn.Parameter(torch.ones([1])*.15))

    def forward(self, x):
        # I = BinarizeIndictator.apply(self.indicator)
        w1 = self.fc1.weight  # + I*self.weight_delta
        x = F.linear(x, w1)
        x = self.act(x)
        x = self.drop(x)
        return x


class freeze_conv(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # weight_shape = self.weight.shape
        # self.register_parameter(name = 'residual', param = torch.nn.Parameter(torch.zeros(weight_shape)))
        # self.register_parameter(name = 'indicator', param = torch.nn.Parameter(torch.ones([1])*INITIALIZATION))
        # self.weight.requires_grad = False
        # self.weight.grad = None

    def forward(self, x):
        w = self.weight  # + I * self.residual
        x = F.conv1d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


files = os.listdir("./1001")
num = 0
histone_name = ['H2AFZ', 'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K9ac', 'H3K9me3', 'H3K27ac', 'H3K27me3', 'H3K36me3',
                'H3K79me2', 'H4K20me1']
cell = '1001'
length = 1001


def train(TrainLoader, net, loss_function, optimizer, device, opts, epoch):
    net.train()
    # net2.train()
    # switch to train mode
    running_loss = 0
    ProgressBar = tqdm(TrainLoader)
    ProgressBar.set_description("Epoch %d" % epoch)
    for i, data in enumerate(ProgressBar, 0):

        inputs, labels = data[0].to(device), data[1].to(device)
        # compute output
        outputs = net(inputs)
        loss = loss_function(outputs.to(device), labels.to(device))
        # measure accuracy and record loss

        ProgressBar.set_postfix(loss=loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()
        if i == 0:
            aver_t = labels.max(1).values  # .max(1).values  signal.mean(1).values.squeeze()
            aver_p = outputs.max(1).values  # outputs.mean(1).values.squeeze()
        else:
            aver_t = torch.cat([aver_t, labels.max(1).values], 0)
            aver_p = torch.cat([aver_p, outputs.max(1).values], 0)

    pcc = np.corrcoef(aver_t.cpu().detach().numpy(), aver_p.cpu().detach().numpy())
    return running_loss / (i + 1), pcc[0][1]


def eval(Val_loader, net, loss_function, device):
    net.eval()
    running_loss = 0
    # switch to train mode
    ProgressBar = tqdm(Val_loader)
    with torch.no_grad():
        for i, data in enumerate(ProgressBar, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            # compute output
            outputs = net(inputs)
            loss = loss_function(outputs.to(device), labels.to(device))
            # measure accuracy and record loss
            running_loss += loss.item()
            if i == 0:
                aver_t = labels.max(1).values  # .max(1).values  signal.mean(1).values.squeeze()
                aver_p = outputs.max(1).values  # outputs.mean(1).values.squeeze()
            else:
                aver_t = torch.cat([aver_t, labels.max(1).values], 0)
                aver_p = torch.cat([aver_p, outputs.max(1).values], 0)
    pcc = np.corrcoef(aver_t.cpu().detach().numpy(), aver_p.cpu().detach().numpy())
    return running_loss / (i + 1), pcc[0][1]


def _finetune(net, train_loader, val_loader, opts: options, name):
    epochs = opts.args.epochs
    init_lr = opts.args.lr
    wd = opts.args.wd
    # gpu = opts.args.gpu
    experiment_path = os.path.join(opts.args.result_path, name)
    if opts.args.result_path and not os.path.exists(opts.args.result_path):
        os.makedirs(opts.args.result_path)

    if not (os.path.exists(experiment_path)):
        os.makedirs(experiment_path)
    criterion = nn.MSELoss().cuda(device)
    optimizer = optim.Adam(net.parameters(), init_lr,
                           weight_decay=wd)
    best_val_loss = 100
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    val_errs = []
    for epoch in range(0, epochs + 1):
        train_loss, train_pcc = train(train_loader, net, criterion, optimizer, device, opts,
                                                                epoch)
        print(train_loss, train_pcc)
        if epoch % opts.args.eval_epochs == 0:
            # evaluate the performance of initialization
            val_loss, val_pcc = eval(val_loader, net, criterion, device)
            print(val_loss, val_pcc)
            val_errs.append(val_loss)

            is_best = val_loss < best_val_loss
            best_val_loss = min(val_loss, best_val_loss)
            state = {
                'epoch': epoch,
                'train_PCC':train_pcc,
                'train_Loss': train_loss,
                'val_PCC': val_pcc,
                'val_loss': val_loss,
                'state_dict_net': net.module.state_dict() if opts.args.multi_gpu else net.state_dict(),
            }
            if is_best:
                best_loss = val_loss
                best_pcc = val_pcc
                best_model_path = ('%s/model_best.pth' % experiment_path)
                torch.save(state, best_model_path)

            val_path = os.path.join(experiment_path, 'val_err')
            # np.save(val_path, val_errs)

        scheduler.step()
    # file = open('./results/human_DNA_CNN', "a")
    # file.write(name + " " + str(np.round(best_auc, 5)) + " " + str(np.round(best_acc, 5)) + " " + str(np.round(best_prc, 5)) +"\n")
    # file.close()


def get_dataset(species):
    i = 0
    for name in files:
        X_1 = np.load("./1001/" + name + "/X.pt.npy")
        signal_1 = np.load("./1001/" + name + "/TF_signal.pt.npy")
        signal_1 = np.log2(signal_1 + 1)
        if i == 0:
            X = X_1
            Activity = signal_1
            i += 1
        else:
            X = np.append(X,X_1,0)
            Activity = np.append(Activity,signal_1,0)
    l = len(X)
    shuffle_ix = np.random.permutation(np.arange(len(X)))
    X = X[shuffle_ix]
    Activity = Activity[shuffle_ix]
    number = len(X) // 10
    X_train = X[0:8 * number]
    Activity_train = Activity[0:8 * number]
    X_test = X[9 * number:10 * number]
    Activity_test = Activity[9 * number:10 * number]
    X_validation = X[8 * number:9 * number]
    Activity_validation = Activity[8 * number:9 * number]
    X_test = torch.from_numpy(X_test)
    Activity_test = torch.from_numpy(Activity_test)
    X_train = torch.from_numpy(X_train)
    Activity_train = torch.from_numpy(Activity_train)
    X_validation = torch.from_numpy(X_validation)
    Activity_validation = torch.from_numpy(Activity_validation)

    TrainLoader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(X_train, Activity_train),
                                              # , histone_train, TF_signal_train
                                              batch_size=64, shuffle=True, num_workers=0)
    TestLoader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(X_test, Activity_test),
                                             # , histone_test, TF_signal_test
                                             batch_size=64, shuffle=True, num_workers=0)
    ValidationLoader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(X_validation, Activity_validation),
                                                   # , histone_validation, TF_signal_validation
                                                   batch_size=64, shuffle=True, num_workers=0)
    return TrainLoader, TestLoader, ValidationLoader


# mus_tissues = ['brain','heart','kidney','lung','spleen']
# fry_tissues = ['Blastoderm','brain','neuron', 'Ovary', 'Pupae']

if __name__ == "__main__":
    #train and validate
    opts = options()
    files = os.listdir("./1001")
    train_loader, test_loader, val_loader = get_dataset('human')
    net = HEAP_DNA()
    net.to(device)
    _finetune(net, train_loader, val_loader, opts, 'human')
    opts.log_settings()

    #test
    experiment_path = os.path.join(opts.args.result_path, 'human')
    pretrained_dict = torch.load(experiment_path + '/model_best.pth')
    net = HEAP_DNA()
    net.to(device)
    net.load_state_dict(pretrained_dict['state_dict_net'])
    criterion = nn.MSELoss().cuda(device)
    eval(test_loader, net, criterion, device)