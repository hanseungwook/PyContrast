import torch
import torch.nn as nn
from .resnet import model_dict
from .util import Normalize, JigsawHead


class RGBSingleHead(nn.Module):
    """RGB model with a single linear/mlp projection head"""
    def __init__(self, name='resnet50', head='linear', feat_dim=128, sup_mode=''):
        super(RGBSingleHead, self).__init__()

        name, width = self._parse_width(name)
        dim_in = int(2048 * width)
        self.width = width

        self.encoder = model_dict[name](width=width)

        if head == 'linear':
            self.head = nn.Sequential(
                nn.Linear(dim_in, feat_dim),
                Normalize(2)
            )
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim),
                Normalize(2)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))
        
        self.online_clf = nn.Linear(dim_in, 1000)

    @staticmethod
    def _parse_width(name):
        if name.endswith('x4'):
            return name[:-2], 4
        elif name.endswith('x2'):
            return name[:-2], 2
        else:
            return name, 1

    def forward(self, x, mode=0):
        # mode --
        # 0: normal encoder,
        # 1: momentum encoder,
        # 2: testing mode,
        # 3: online classifier mode
        feat = self.encoder(x)
        if mode == 0 or mode == 1:
            feat = self.head(feat)
        
        elif mode == 3:
            feat_proj = self.head(feat)
            feat_clf = self.online_clf(feat.detach())

            return feat_proj, feat_clf
            
        return feat


    def forward_online_clf(self, x):
        return self.online_clf(x.detach())


class RGBMultiHeads(RGBSingleHead):
    """RGB model with Multiple linear/mlp projection heads"""
    def __init__(self, name='resnet50', head='linear', feat_dim=128):
        super(RGBMultiHeads, self).__init__(name, head, feat_dim)

        self.head_jig = JigsawHead(dim_in=int(2048*self.width),
                                   dim_out=feat_dim,
                                   head=head)

    def forward(self, x, x_jig=None, mode=0):
        # mode --
        # 0: normal encoder,
        # 1: momentum encoder,
        # 2: testing mode
        if mode == 0:
            feat = self.head(self.encoder(x))
            feat_jig = self.head_jig(self.encoder(x_jig))
            return feat, feat_jig
        elif mode == 1:
            feat = self.head(self.encoder(x))
            return feat
        else:
            feat = self.encoder(x)
            return feat


class CMCSingleHead(nn.Module):
    """CMC model with a single linear/mlp projection head"""
    def __init__(self, name='resnet50', head='linear', feat_dim=128):
        super(CMCSingleHead, self).__init__()

        name, width = self._parse_width(name)
        dim_in = int(2048 * width)
        self.width = width

        self.encoder1 = model_dict[name](width=width, in_channel=1)
        self.encoder2 = model_dict[name](width=width, in_channel=2)

        if head == 'linear':
            self.head1 = nn.Sequential(
                nn.Linear(dim_in, feat_dim),
                Normalize(2)
            )
            self.head2 = nn.Sequential(
                nn.Linear(dim_in, feat_dim),
                Normalize(2)
            )
        elif head == 'mlp':
            self.head1 = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim),
                Normalize(2)
            )
            self.head2 = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim),
                Normalize(2)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    @staticmethod
    def _parse_width(name):
        if name.endswith('x4'):
            return name[:-2], 2
        elif name.endswith('x2'):
            return name[:-2], 1
        else:
            return name, 0.5

    def forward(self, x, mode=0):
        # mode --
        # 0: normal encoder,
        # 1: momentum encoder,
        # 2: testing mode
        x1, x2 = torch.split(x, [1, 2], dim=1)
        feat1 = self.encoder1(x1)
        feat2 = self.encoder2(x2)
        if mode == 0 or mode == 1:
            feat1 = self.head1(feat1)
            feat2 = self.head2(feat2)
        return torch.cat((feat1, feat2), dim=1)


class CMCMultiHeads(CMCSingleHead):
    """CMC model with Multiple linear/mlp projection heads"""
    def __init__(self, name='resnet50', head='linear', feat_dim=128):
        super(CMCMultiHeads, self).__init__(name, head, feat_dim)

        self.head1_jig = JigsawHead(dim_in=int(2048*self.width),
                                    dim_out=feat_dim,
                                    head=head)
        self.head2_jig = JigsawHead(dim_in=int(2048*self.width),
                                    dim_out=feat_dim,
                                    head=head)

    def forward(self, x, x_jig=None, mode=0):
        # mode --
        # 0: normal encoder,
        # 1: momentum encoder,
        # 2: testing mode
        x1, x2 = torch.split(x, [1, 2], dim=1)
        feat1 = self.encoder1(x1)
        feat2 = self.encoder2(x2)

        if mode == 0:
            x1_jig, x2_jig = torch.split(x_jig, [1, 2], dim=1)
            feat1_jig = self.encoder1(x1_jig)
            feat2_jig = self.encoder2(x2_jig)

            feat1, feat2 = self.head1(feat1), self.head2(feat2)
            feat1_jig = self.head1_jig(feat1_jig)
            feat2_jig = self.head2_jig(feat2_jig)
            feat = torch.cat((feat1, feat2), dim=1)
            feat_jig = torch.cat((feat1_jig, feat2_jig), dim=1)
            return feat, feat_jig
        elif mode == 1:
            feat1, feat2 = self.head1(feat1), self.head2(feat2)
            return torch.cat((feat1, feat2), dim=1)
        else:
            return torch.cat((feat1, feat2), dim=1)


NAME_TO_FUNC = {
    'RGBSin': RGBSingleHead,
    'RGBMul': RGBMultiHeads,
    'CMCSin': CMCSingleHead,
    'CMCMul': CMCMultiHeads,
}


def build_model(opt):
    # specify modal key
    branch = 'Mul' if opt.jigsaw else 'Sin'
    model_key = opt.modal + branch

    model = NAME_TO_FUNC[model_key](opt.arch, opt.head, opt.feat_dim, opt.sup_mode)
    if opt.mem == 'moco':
        model_ema = NAME_TO_FUNC[model_key](opt.arch, opt.head, opt.feat_dim, opt.sup_mode)
    else:
        model_ema = None

    return model, model_ema
