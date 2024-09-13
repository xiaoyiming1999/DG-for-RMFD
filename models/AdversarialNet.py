from torch import nn


def grl_hook():
    def fun1(grad):
        return -1.0 * grad.clone()
    return fun1


class Discriminator(nn.Module):
    def __init__(self, output_size):
        super(Discriminator, self).__init__()
        self.ad_layer1 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.ad_layer2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.ad_layer3 = nn.Linear(1024, output_size)

    def forward(self, x):

        x = x * 1.0
        x.register_hook(grl_hook())
        x = self.ad_layer1(x)
        x = self.ad_layer2(x)
        y = self.ad_layer3(x)
        return y
