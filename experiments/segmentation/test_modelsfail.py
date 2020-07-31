import importlib
print('import importlib')
import torch
print('import torch')
import encoding
print('import encoding')
# from option import Options
print('from option import Options')
from torch.autograd import Variable
from test import Options
from encoding.models import get_model, get_segmentation_model, MultiEvalModule
from encoding.nn import SegmentationLosses, SyncBatchNorm
import cv2
import numpy as np
from torchvision import transforms
import torchvision

norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
print('begin')
if __name__ == "__main__":
    print('please write the parse')
    args = Options().parse()
    print('i am here')
    pretrained = args.resume is None and args.verify is None
    if args.model_zoo is not None:
        model = get_model(args.model_zoo, pretrained=pretrained)
        model.base_size = args.base_size
        model.crop_size = args.crop_size
    else:
        model = get_segmentation_model(args.model, dataset=args.dataset,
                                       backbone=args.backbone, aux = args.aux,
                                       se_loss=args.se_loss,
                                       norm_layer=torch.nn.BatchNorm2d if args.acc_bn else SyncBatchNorm,
                                       base_size=args.base_size, crop_size=args.crop_size)
    # model = encoding.models.get_segmentation_model(args.model, dataset=args.dataset, aux=args.aux,
    #                                                backbone=args.backbone,
    #                                                se_loss=args.se_loss, norm_layer=torch.nn.BatchNorm2d)
    print('Creating the model:')
    
    print(model)
    model.cuda()
    model.eval()
    
    # x = Variable(torch.Tensor(4, 3, 480, 480)).cuda()
    
    rgb = cv2.imread('example.jpg')
    rgb = np.transpose(rgb, (2, 0, 1))
    rgb = norm(torch.from_numpy(rgb.astype(np.float32)))
    rgb = Variable(rgb).cuda()
    # print(rgb.shape())

    with torch.no_grad():
        # out = model(rgb)
        semantic = model(rgb.unsqueeze(0))
        # print(semantic)
        for s in semantic:

            pred = torch.max(s,1)[1].cpu()
            # pred = pred.numpy()
            # pred = pred *255
            print(type(pred))
            torchvision.utils.save_image(pred, './test_mask.png')
            pred = np.transpose(pred, (1, 2, 0))
            # image_np = cv2.imdecode(pred, cv2.IMREAD_COLOR)
            # cv2.imshow('iw', image_np)
            # cv2.waitKey(0)

#    for y in out:
#        print(y.size())
