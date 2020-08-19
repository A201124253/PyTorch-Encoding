#!/usr/bin/env python
import sys
# sys.path.insert(0,'/home/lzj/anaconda3/envs/materialseg/lib/python3.7/site-packages')
# sys.path.insert(0,'/home/lzj/anaconda3/lib/python3.7/site-packages')
# print('\n'.join(sys.path))
import cv2
# for rospy
import rospy
# for String and Image
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage	
# for image
import message_filters
# import cv_bridge
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transform

import encoding



# input_transform
input_transform = transform.Compose([
    transform.ToTensor(),
    transform.Normalize([.485, .456, .406], [.229, .224, .225])])

# Get the model0,0,0
# model = encoding.models.get_model('fcn_resnest50_ade', pretrained=True).cuda()
# model = encoding.models.get_model('deeplab_resnest50_minc', pretrained=True).cuda()
# model = encoding.models.get_model('fcn_resnest50_minc', pretrained=True).cuda()
model = encoding.models.get_model('deeplab_resnest101_minc', pretrained=True).cuda()

# print(model)
model.eval()
SAVENAME = 'deeplab_resnest101_108'
DATE = '202008181233'
mincpallete = [0,0,0, 0,192,0,	129,0,0,	128,128,0,	3,65,142,	0,0,127,	127,0,127,	0,128,128,	150,150,150,	65,0,0,	192,0,129,	65,129,0,	255,255,255,	65,0,128,	100,70,50,	50,65,100, 200,127,129,	0,65,0,	129,65,0,	8,250,8,	127,192,0,	0,65,129,	65,128,128,	192,127,0]

class image_seg:

    def __init__(self):
        self.image_pub = rospy.Publisher("material_segmentation", CompressedImage, queue_size=1)
        #print("subscriber")
        self.image_sub = rospy.Subscriber('/camera/color/image_raw/compressed', CompressedImage, self.callback, queue_size = 1)
        
    
    def callback(self, ros_data):
        
        np_arr = np.frombuffer(ros_data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        # cv2.imwrite(SAVENAME + DATE+'_orig_'+'.png',image_np)
        # convert it to pil format 
        im = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(im)
        # im_pil = im_pil.convert('RGB')
        
        transform = input_transform
        im_pil = transform(im_pil)
        # print(type(img))
        
        # Make prediction 
        im_pil = im_pil.cuda().unsqueeze(0)
        output = model.evaluate(im_pil)
        predict = torch.max(output, 1)[1].cpu().numpy() + 1
	    # Get color pallete for visualization
        mask = encoding.utils.get_mask_pallete(predict, 'minc_dataset')
        # print(type(mask))
        # print(mask)
        # mask.save(SAVENAME + DATE +'_rgb_'+'.png')
        # mask.show()
        
        # time.sleep(5)

        # mask.close()

        pil2ros = np.asarray(mask)
        
        pil2ros3_shape = np.insert(pil2ros.shape,2,[3])
        
        pil2ros3 = np.zeros((pil2ros3_shape)).astype("uint8")
        pil2ros3[:,:,0] = pil2ros
        pil2ros3[:,:,1] = pil2ros
        pil2ros3[:,:,2] = pil2ros
        
        for i in range(24):
            pil2ros3 = np.where(pil2ros3==((i,i,i)),((mincpallete[i*3+2],mincpallete[i*3+1],mincpallete[i*3])),pil2ros3)
        pil2ros3 = np.array(pil2ros3, dtype=np.uint8)
        cv2.imshow("window",pil2ros3)
        k = cv2.waitKey(2)
        '''
        pil2ros = cv2.cvtColor(pil2ros, cv2.COLOR_RGB2BGR)
        cv2.imwrite(SAVENAME + DATE+'_gray_'+'.png',pil2ros)
        # print(pil2ros.shape)
        
        # show the mask
        cv2.imshow("Image window", pil2ros)
        cv2.waitKey(3)
        '''
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', pil2ros3)[1]).tostring()
        # print('i am publisher')
        self.image_pub.publish(msg)


def main(args):
    
    topic = 'material_seg_result'
    pub = rospy.Publisher(topic, String, queue_size=1)
    rospy.init_node('material_seg', anonymous= True)
    rospy.loginfo("I will publish to the topic %s", topic)
    
    while not rospy.is_shutdown():
        '''
        str = "hello world %s"%rospy.get_time()
        rospy.loginfo("i am here")
        pub.publish(str)
        rospy.sleep(0.1)
        '''
        # rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        ims = image_seg()
        rospy.init_node('material_seg', anonymous= True)
        try:
            rospy.spin()
            # print("behind spin")
        except KeyboardInterrupt:
            print("Shutting down")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)

