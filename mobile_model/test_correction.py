import cv2
import matplotlib.pyplot as plt
import imutils
import  numpy as np
from imutils.perspective import  four_point_transform
from imutils import auto_canny, contours
from skimage import data, exposure
from PIL import Image,ImageStat



#答题卡矫正



def adaptiveThreshold(img, sub_thresh=0.15):
    image = cv2.imread(img)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #    计算积分图像
    integralimage = cv2.integral(gray_image, cv2.CV_32F)

    width = gray_image.shape[1]
    height = gray_image.shape[0]
    win_length = int(width / 10)
    image_thresh = np.zeros((height, width, 1), dtype=np.uint8)
    #    perform threshholding
    for j in range(height):
        for i in range(width):
            x1 = i - win_length
            x2 = i + win_length
            y1 = j - win_length
            y2 = j + win_length

            #            check the border
            if (x1 < 0):
                x1 = 0
            if (y1 < 0):
                y1 = 0
            if (x2 > width):
                x2 = width - 1
            if (y2 > height):
                y2 = height - 1
            count = (x2 - x1) * (y2 - y1)

            #            I(x,y) = s(x2,y2) - s(x1,y2) - s(x2, y1) + s(x1, y1)
            sum = integralimage[y2, x2] - integralimage[y1, x2] - integralimage[y2, x1] - integralimage[y1, x1]
            if (int)(gray_image[j, i] * count) < (int)(sum * (1.0 - sub_thresh)):
                image_thresh[j, i] = 0
            else:
                image_thresh[j, i] = 255

    return image_thresh





def image_correction(img_path):
    '''
    :param img_path: The path of a single image
    :return: correction image , type :numpy array
    '''
    image = cv2.imread(img_path)
    newimage = image.copy()
    #转换为灰度图像
    image= exposure.adjust_gamma(image, 0.5)  #亮度增强
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #高斯滤波
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    #自适应二值化方法
    blurred=cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,51,2)
    # ret1, blurred = cv2.threshold(blurred, 0, 255, cv2.THRESH_OTSU)  #OTSU二值化

    '''
    adaptiveThreshold函数：第一个参数src指原图像，原图像应该是灰度图。
        第二个参数x指当像素值高于（有时是小于）阈值时应该被赋予的新的像素值
        第三个参数adaptive_method 指： CV_ADAPTIVE_THRESH_MEAN_C 或 CV_ADAPTIVE_THRESH_GAUSSIAN_C
        第四个参数threshold_type  指取阈值类型：必须是下者之一  
                                     •  CV_THRESH_BINARY,
                            • CV_THRESH_BINARY_INV
         第五个参数 block_size 指用来计算阈值的象素邻域大小: 3, 5, 7, ...
        第六个参数param1    指与方法有关的参数。对方法CV_ADAPTIVE_THRESH_MEAN_C 和 CV_ADAPTIVE_THRESH_GAUSSIAN_C， 它是一个从均值或加权均值提取的常数, 尽管它可以是负数。
    '''
    #这一步可有可无，主要是增加一圈白框，以免刚好卷子边框压线后期边缘检测无果。好的样本图就不用考虑这种问题
    # blurred=cv2.copyMakeBorder(blurred,5,5,5,5,cv2.BORDER_CONSTANT,value=(255,255,255))
    #2。边缘检测部分
    edged = cv2.Canny(blurred, 35, 189)

    # 从边缘图中寻找轮廓，然后初始化答题卡对应的轮廓
    '''
    findContours
    image -- 要查找轮廓的原图像
    mode -- 轮廓的检索模式，它有四种模式：
         cv2.RETR_EXTERNAL  表示只检测外轮廓                                  
         cv2.RETR_LIST 检测的轮廓不建立等级关系
         cv2.RETR_CCOMP 建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。如果内孔内还有一个连通物体，
                  这个物体的边界也在顶层。
         cv2.RETR_TREE 建立一个等级树结构的轮廓。
    method --  轮廓的近似办法：
         cv2.CHAIN_APPROX_NONE 存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max （abs (x1 - x2), abs(y2 - y1) == 1
         cv2.CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需
                           4个点来保存轮廓信息
          cv2.CHAIN_APPROX_TC89_L1，CV_CHAIN_APPROX_TC89_KCOS使用teh-Chinl chain 近似算法
    '''
    cnts = cv2.findContours(edged, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)    #返回三个参数，iamge，counts，和位置信息,

    cnts = cnts[0] if imutils.is_cv2() else cnts[1]  #判断cnt[0]是否是轮廓列表，opencv2中cnt【0】为列表信息，opencv3中cnts【1】为 列表信息。
    # 确保至少有一个轮廓被找到
    if len(cnts) > 0:
        # 将轮廓按大小降序排序
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        # 对排序后的轮廓循环处理

        four_point=[]   #返回的四点
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)  # (x,y)是rectangle的左上角坐标， (w,h)是width和height
            area = cv2.contourArea(c)
            c=w/h    #获取区域的长宽比例。
            if area<200 or c<0.7 or c>1.3:    #阈值自己设定
                pass
            else:
                one_area=blurred[:,x:x + w][y:y + h,:]   #获取出来符合条件的区域
                all_pixel=[]
                for ii in range(one_area.shape[0]):
                    for jj in range(one_area.shape[1]):
                        if one_area[ii][jj]==255:
                            all_pixel.append(0)
                        else:
                            all_pixel.append(1)
                all_pixel=np.mean(np.array(all_pixel))
                if all_pixel <0.8:   #设定阈值 0.8保证识别出来的二维码区域不被包含
                    pass
                else:
                    # img = cv2.rectangle(newimage, (x, y), (x + w, y + h), (0, 255, 0), 1)

                    if [x+w/2,y+h/2] in four_point:
                        pass
                    else:
                        four_point.append([x + w / 2, y + h / 2])
        if len(four_point)==4:
            warped = four_point_transform(newimage, np.array(four_point))


        return warped



#客观题区、主观题区域提取

def object_area(c_img):

    '''
    :param c_img: A single corrected image,numpy array
    :return:Objective subject area image and Subject area image,type :numpy array
    '''
    img=Image.fromarray(np.uint8(c_img))
    img=img.resize((2100,2970))

    #客观题ROI
    objective_ROI=(0,880,2100,2970)
    img=img.crop((objective_ROI))
    c_img=np.array(img)

    gray = cv2.cvtColor(c_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    blurred = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 2)
    edged = cv2.Canny(blurred,10, 100)

    # 从边缘图中寻找轮廓，然后初始化答题卡对应的轮廓
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    if len(cnts) > 0:
        # 将轮廓按大小降序排序
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        rectangular_area=[]
        new_set=[]
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)  # (x,y)是rectangle的左上角坐标， (w,h)是width和height
            rectangular_area.append(w*h)   #区域面积
            new_set.append([x, y, w, h])
        set_dict=dict(zip(rectangular_area,new_set))
        rectangular_area=sorted(rectangular_area, reverse=True)
        set_point=[]
        for s in rectangular_area:
            new_x1,new_y1,new_x2,new_y2=set_dict.get(s)[0],set_dict.get(s)[1],set_dict.get(s)[0] + set_dict.get(s)[2], set_dict.get(s)[1] + set_dict.get(s)[3]
            if new_x2-new_x1<1800 or new_y2-new_y1<200:
                pass
            else:
                # img = cv2.rectangle(c_img, (new_x1, new_y1),
                #                     (new_x2, new_y2), (0, 255, 0), 3)
                set_point.append([new_x1,new_y1,new_x2,new_y2])
        flag=min([x[1] for x in set_point])
        set_point1=[]
        set_point2=[]

        for b in set_point:
            if flag-10<= b[1] <=flag+10:
                set_point1.append(b)
            else:
                set_point2.append(b)

        flag1=min([x[1] for x in set_point2])
        set_point3=[]
        for b1 in set_point2:
            if flag1 - 10 <= b1[1] <= flag1 + 10:
                set_point3.append(b1)
            else:
                pass
        new_rectangular_area=[]
        if len(set_point3)==0 and len(set_point1) !=0:
            point_1_x1=int(np.mean([x[0] for x in set_point1]))
            point_1_y1=int(np.mean([x[1] for x in set_point1]))
            point_1_x2=int(np.mean([x[2] for x in set_point1]))
            point_1_y2=int(np.mean([x[3] for x in set_point1]))

            new_rectangular_area.append([point_1_x1,point_1_y1,point_1_x2,point_1_y2])
        else:
            point_1_x1 = int(np.mean([x[0] for x in set_point1]))
            point_1_y1 = int(np.mean([x[1] for x in set_point1]))
            point_1_x2 = int(np.mean([x[2] for x in set_point1]))
            point_1_y2 = int(np.mean([x[3] for x in set_point1]))

            point_2_x1 = int(np.mean([x[0] for x in set_point3]))
            point_2_y1 = int(np.mean([x[1] for x in set_point3]))
            point_2_x2 = int(np.mean([x[2] for x in set_point3]))
            point_2_y2 = int(np.mean([x[3] for x in set_point3]))



            new_rectangular_area.append([point_1_x1, point_1_y1, point_1_x2, point_1_y2])
            new_rectangular_area.append([point_2_x1, point_2_y1, point_2_x2, point_2_y2])
        #画出识别出来的区域以便观察
        # for ss in new_rectangular_area:
        #     img = cv2.rectangle(c_img, (ss[0], ss[1]),
        #                         (ss[2], ss[3]), (0, 255, 0), 2)

        ss0=new_rectangular_area[0]
        ob_img = c_img[:, ss0[0]:ss0[2]][ss0[1]:ss0[3], :]
        if len(new_rectangular_area)==2:
            ss1 = new_rectangular_area[1]
            else_area = c_img[:, ss1[0]:ss1[2]][ss1[1]:ss1[3], :]
        else:
            else_area=[]

        return ob_img,else_area


# # c_img=image_correction('/Users/wywy/Desktop/ob_test/ob_test6.jpg')
# # ob_img,else_area=object_area(c_img)
# img=cv2.imread('/Users/wywy/Desktop/test5.jpg')
# # gray = cv2.cvtColor(ob_img, cv2.COLOR_BGR2GRAY)
# thresh=adaptiveThreshold(img)
#
#
# ChQImg = cv2.blur(thresh, (10, 10))
#
# ChQImg = cv2.threshold(ChQImg, 100, 225, cv2.THRESH_BINARY)[1]
#
# cv2.imwrite('/Users/wywy/Desktop/ob_test6.jpg',ChQImg)
# # cv2.imshow('test',ChQImg)
# # cv2.waitKey(0)
#

def brightness(im_file):
    im = Image.open(im_file).convert('L')
    stat = ImageStat.Stat(im)
    return stat.mean[0]


a=brightness('/Users/wywy/Desktop/ob_test/0.jpg')
org_img=cv2.imread('/Users/wywy/Desktop/test/3/1537164784453.jpg')
print(org_img.shape)
gary_img=cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
blurred=cv2.adaptiveThreshold(gary_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)


cv2.imshow('test',blurred)
cv2.waitKey(0)





