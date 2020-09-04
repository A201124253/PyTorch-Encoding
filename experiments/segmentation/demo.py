import torch
import sys
# sys.path.insert(0,'/home/lzj/anaconda3/envs/materialseg/lib/python3.7/site-packages')

import encoding
print('encoding')
# Get the model
model = encoding.models.get_model('fcn_resnest50_ade', pretrained=True).cuda()

model.eval()

# Prepare the image
url = 'https://github.com/zhanghang1989/image-data/blob/master/' + \
      'encoding/segmentation/ade20k/ADE_val_00001142.jpg?raw=true'
filename = 'example.jpg'
img = encoding.utils.load_image(
    encoding.utils.download(url, filename)).cuda().unsqueeze(0)

# Make prediction
output = model.evaluate(img)
print("output shape is")
print(output.shape)
predict = torch.max(output, 1)[1].cpu().numpy() + 1
print("predict.shape is")
print(predict.shape)
print(predict)
# Get color pallete for visualization
mask = encoding.utils.get_mask_pallete(predict, 'ade20k')
mask.save('output.png')
