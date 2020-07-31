import torch
import encoding
from PIL import Image
import numpy as np
import cv2

# Get the model
model = encoding.models.get_model('fcn_resnest50_ade', pretrained=True).cuda()
model.eval()

# Prepare the image
url = 'https://github.com/zhanghang1989/image-data/blob/master/' + \
      'encoding/segmentation/ade20k/ADE_val_00001142.jpg?raw=true'
filename = 'example.jpg'
# img = encoding.utils.load_image(
#     encoding.utils.download(url, filename)).cuda().unsqueeze(0)

# PIL to opencv
'''
im = Image.open(filename).convert('RGB')
im.save('im.png')
a = np.asarray(im)
print(type(im))
print(type(a))
'''
# opencv to PIL

im = cv2.imread(filename)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
im_pil = Image.fromarray(im)
im_pil = im_pil.convert('RGB')
im_pil.save("bgr2rgb.png")


im_cv = np.asarray(im_pil)
im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

cv2.imwrite('cvformat.png',im)



# img = encoding.utils.load_image(filename).cuda().unsqueeze(0)
# print(type(img))
# print(img.cpu().numpy())
# print(type(img))
# Make prediction
# output = model.evaluate(img)
# predict = torch.max(output, 1)[1].cpu().numpy() + 1

# Get color pallete for visualization
# mask = encoding.utils.get_mask_pallete(predict, 'ade20k')
# print(type(mask))
# mask.save('output.png')
