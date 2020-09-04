import torch
import encoding
import cv2
import os

# Get the model
modelname='deeplab_resnest101_minc'
# modelname='deeplab_resnet50s_minc'
# modelname='deeplab_resnest50_minc'
# modelname='fcn_resnest50_minc'

# transfer
modelname='fcn_resnet50_minc'

# model = encoding.models.get_model('deeplab_resnest101_minc', pretrained=True)
# model = encoding.models.get_model('deeplab_resnet50s_minc', pretrained=True).cuda()
# model = encoding.models.get_model('deeplab_resnest50_minc', pretrained=True).cuda()
model = encoding.models.get_model(modelname, pretrained=True).cuda()

model.eval()

# Prepare the image
# url = 'https://github.com/zhanghang1989/image-data/blob/master/' + \
      # 'encoding/segmentation/ade20k/ADE_val_00001142.jpg?raw=true'
# filename = 'example.jpg'
imagename = '000090727'

img = cv2.imread('images/'+imagename+'.jpg', cv2.IMREAD_UNCHANGED)
 
print('Original Dimensions : ',img.shape) 


scale_percent = 100 # percent of original size
if(img.shape[1]>2000 or img.shape[0]>2000):
    scale_percent = 25

width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
# save image
img = cv2.imwrite('images/'+imagename+'_re.jpg', resized)

img = encoding.utils.load_image('images/'+imagename+'_re.jpg').cuda().unsqueeze(0)

# Make prediction
output = model.evaluate(img)
predict = torch.max(output, 1)[1].cpu().numpy() + 1

# Get color pallete for visualization
mask = encoding.utils.get_mask_pallete(predict, 'minc_dataset')
if not os.path.exists('images/'+ imagename):
    os.makedirs('images/'+ imagename)
mask.save('images/'+ imagename + '/'+ modelname + '.png')
