# In this section, we will apply an CNN to extract features and implement a classification task.
# Firstly, we should build the model by PyTorch. We provide a baseline model here.
# You can use your own model for better performance
import torch.nn as nn
import torch.nn.functional as F

class Doubleconv_33(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Doubleconv_33, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ch_in, ch_out, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch_out, ch_out, kernel_size=3),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class Doubleconv_35(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Doubleconv_35, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ch_in, ch_out, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch_out, ch_out, kernel_size=5),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class Doubleconv_37(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Doubleconv_37, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ch_in, ch_out, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch_out, ch_out, kernel_size=7),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class Tripleconv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Tripleconv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ch_in, ch_out, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch_out, ch_out, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch_out, ch_out, kernel_size=3),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class MLP(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(ch_in, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, ch_out),
        )

    def forward(self, input):
        return self.fc(input)


class Mscnn(nn.Module):
    # TODO: Build a better model
    def __init__(self, ch_in, ch_out):
        super(Mscnn, self).__init__()
        self.conv11 = Doubleconv_33(ch_in, 64)
        self.pool11 = nn.MaxPool1d(3, stride=3)
        self.conv12 = Doubleconv_33(64, 128)
        self.pool12 = nn.MaxPool1d(3, stride=3)
        self.conv13 = Tripleconv(128, 256)
        self.pool13 = nn.MaxPool1d(2, stride=2)
        self.conv14 = Tripleconv(256, 512)
        self.pool14 = nn.MaxPool1d(2, stride=2)
        self.conv15 = Tripleconv(512, 512)
        self.pool15 = nn.MaxPool1d(2, stride=2)

        self.out = MLP(512*36, ch_out)  

    def forward(self, x):
        c11 = self.conv11(x)
        p11 = self.pool11(c11)
        c12 = self.conv12(p11)
        p12 = self.pool12(c12)
        c13 = self.conv13(p12)
        p13 = self.pool13(c13)
        c14 = self.conv14(p13)
        p14 = self.pool14(c14)
        c15 = self.conv15(p14)
        p15 = self.pool15(c15)
        merge = p15.view(p15.size()[0], -1)
        output = self.out(merge)
        output = F.sigmoid(output)
        return output