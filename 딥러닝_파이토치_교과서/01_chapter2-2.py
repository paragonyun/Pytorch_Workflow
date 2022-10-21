import torch.nn as nn

class MLP(nn.Module) :
    def __init__(self) :
        super(MLP, self).__init__()

        self.layer1 = nn.Sequential(

                    nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2)

                    )

        self.layer2 = nn.Sequential(

                    nn.Conv2d(in_channels=64, out_channels=30, kernel_size=5),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2)

                    )

        self.fc_layer = nn.Sequential(

                    nn.Linear(in_features=30*5*5, out_features=10, bias=True),
                    nn.ReLU(inplace=True)

                    )

        self.fin_activation = nn.Softmax(dim=10)

    def forward(self, x) :
        output = self.layer1(x)
        output = self.layer2(output)
        
        output = output.view(output.shape[0], -1) ## size(0)으로 해도 됨

        output = self.fc_layer(output)

        output = self.fin_activation(output)

        return output

    