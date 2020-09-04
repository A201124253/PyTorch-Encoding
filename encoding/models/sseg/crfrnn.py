
import torch
import torch.nn as nn

# from encoding.nn.crfasrnn.filters import SpatialFilter, BilateralFilter
# from encoding.nn.crfasrnn.params import DenseCRFParams
# from encoding.nn.crfasrnn import crfasrnn_model

from encoding.nn.crfasrnn.crfrnn import CrfRnn
# from encoding.models.sseg.fcn import FCNHead
from encoding.models.sseg.deeplab import DeepLabV3
from torch.nn.functional import interpolate


# print('import')

class CrfRnnNet(DeepLabV3):
    """
    The full CRF-RNN network with the FCN-8s backbone as described in the paper:
    Conditional Random Fields as Recurrent Neural Networks,
    S. Zheng, S. Jayasumana, B. Romera-Paredes, V. Vineet, Z. Su, D. Du, C. Huang and P. Torr,
    ICCV 2015 (https://arxiv.org/abs/1502.03240).
    """

    def __init__(self, nclass, backbone, transfer=True, **kwargs):
        print('here is 1')
        super(CrfRnnNet, self).__init__(nclass=nclass, backbone=backbone,aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, transfer=False, **kwargs)
        if transfer:
            i=0            
            for p in self.parameters():
                p.requires_grad = False
                i+=1
            print("parameter number of no grad is")
            print(i)
        self.crfrnn = CrfRnn(num_labels=nclass, num_iterations=10)
        print('here is 2')
        i=0            
        for p in self.parameters():
            i+=1
            # if i>289:
            #     print(p)
        print("parameter number is")
        print(i)

    def forward(self, image):
        _, _, h, w = image.size()
        c1, c2, c3, c4 = self.base_forward(image)

        outputs = []
        x = self.head(c4)
        x = self.crfrnn(image, x)

        x = interpolate(x, (h,w), **self._up_kwargs)
        outputs.append(x)

        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = interpolate(auxout, (h,w), **self._up_kwargs)
            outputs.append(auxout)
        return tuple(outputs)

crn = CrfRnnNet(nclass=23, backbone='resnest50')
# print('create')


def get_crfrnn(dataset='minc_seg', backbone='resnest50', pretrained=False,
            root='~/.encoding/models', **kwargs):
    # infer number of classes
    from ...datasets import datasets, acronyms
    model = CrfRnnNet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('crfrnn_%s_%s'%(backbone, acronyms[dataset]), root=root)))
    return model

def get_crfrnn_resnest50_minc(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""DeepLabV3 model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_deeplab_resnet50_ade(pretrained=True)
    >>> print(model)
    """
    return get_crfrnn('minc_dataset', 'resnest50', pretrained, root=root, **kwargs)