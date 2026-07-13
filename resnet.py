from torch import nn


class MyResnet (nn.Module):

    def __init__(self, resnet, num_class, neurons_reducer_block=256, freeze_conv=False, p_dropout=0.5, n_feat_conv=2048):

        super(MyResnet, self).__init__()

        self.features = nn.Sequential(*list(resnet.children())[:-1])

        if freeze_conv:
            for param in self.features.parameters():
                param.requires_grad = False

        if neurons_reducer_block > 0:
            self.reducer_block = nn.Sequential(
                nn.Linear(n_feat_conv, neurons_reducer_block),
                nn.BatchNorm1d(neurons_reducer_block),
                nn.ReLU(),
                nn.Dropout(p=p_dropout)
            )
        else:
            self.reducer_block = None

        if neurons_reducer_block > 0:
            self.classifier = nn.Linear(neurons_reducer_block, num_class)
        else:
            self.classifier = nn.Linear(n_feat_conv, num_class)


    def forward(self, img):

        x = self.features(img)

        x = x.view(x.size(0), -1) 
        if self.reducer_block is not None:
            x = self.reducer_block(x)  


        return self.classifier(x)