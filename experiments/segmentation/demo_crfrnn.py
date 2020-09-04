import torch
import sys
# sys.path.insert(0,'/home/lzj/anaconda3/envs/materialseg/lib/python3.7/site-packages')
from crfasrnn.crfasrnn_model import CrfRnnNet
import encoding
from PIL import Image
import torchvision.transforms as transform
from crfasrnn import util
import numpy as np
print('encoding')
# Get the model
# model = encoding.models.get_model('fcn_resnest50_ade', pretrained=True).cuda()
saved_weights_path = "crfasrnn_weights.pth"
model = CrfRnnNet()
model.load_state_dict(torch.load(saved_weights_path))
model.eval()

input_transform = transform.Compose([
    transform.ToTensor(),
    transform.Normalize([.485, .456, .406], [.229, .224, .225])])

# Prepare the image
url = 'https://github.com/zhanghang1989/image-data/blob/master/' + \
      'encoding/segmentation/ade20k/ADE_val_00001142.jpg?raw=true'
filename = 'example.jpg'
# img = encoding.utils.load_image(
#     encoding.utils.download(url, filename)).unsqueeze(0)
img_data, img_h, img_w, size = util.get_preprocessed_image(filename)
print(img_data)
print(img_data.shape)
img_data2 = Image.open(filename)
img_data2 = np.asarray(img_data2, dtype="int32" )
print(img_data2)
# transform = input_transform
# img_data = transform(img_data)
# img_data = img_data.unsqueeze(0)

# Make prediction
output = model(torch.from_numpy(img_data))
print("outputshape is")
print(output.shape)
print(output)
print(torch.max(output, 1)[1].shape)
predict = torch.max(output, 1)[1].cpu().numpy() + 1
print(predict)
# Get color pallete for visualization
mask = encoding.utils.get_mask_pallete(predict, 'ade20k')
mask.save('output.png')
