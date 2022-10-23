import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

class ResNet18 :
    def __init__(self) :
        self.model = models.resnet18(pretrained=True)

    def _param_frozen(self, model) :
        print('Freezing Parameters...')
        
        model = self.model

        for param in model.parameters() :
            param.requires_grad = False
        
        return model

    def _add_fc(self, model) :
        print('Adding Additional FC Layer...')

        model.fc = nn.Linear(512,2)
        for param in model.fc.parameters() :
            param.requires_grad = True

        return model

    def _show_model(self, model) :
        print(summary(model, (3, 224,224)))

    def run(self) :
        model = self._param_frozen(self.model)
        model = self._add_fc(model)
        self._show_model(model)

        return model

'''
resnet = ResNet18()
model = resnet.run()
'''