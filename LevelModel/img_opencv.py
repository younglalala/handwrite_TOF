import cv2
import numpy as np

import urllib.request

def set_pasre(img_path,save_path):
    '''
    答题卡坐标点定位。
    :param
    :return:
    '''
    cc = 0
    image = cv2.imread(img_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image_gray = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # image_gray = cv2.morphologyEx(image_gray, cv2.MORPH_OPEN, kernel)
    _, contours, _ = cv2.findContours(image_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_crop=image.copy()
    cv2.drawContours(img_crop, contours, -1, (0, 255, 0), 3)
    squares = []
    epsilon = 0.1
    min_area = 50*50    #根据实际情况调整最小区域面积
    max_area = 70*70    #调整区域最大值
    for cnt in contours:
        cnt_len = cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, epsilon * cnt_len, True)
        if len(cnt) != 4:
            continue

        area = cv2.contourArea(cnt)

        if not min_area<area<max_area:
            continue
        squares.append(cnt.reshape(-1, 2))
        cv2.drawContours(img_crop, contours, -1, (0, 0, 0), 3)
    print(len(squares))
    points = []
    for square in squares:
        #对坐标进行排序
        points.append([square[0],square[1], square[2],square[3]])
        x_set = np.array([square[0],square[1], square[2],square[3]])[:, 0]
        y_set = np.array([square[0],square[1], square[2],square[3]])[:, 1]
        np.array(x_set.sort())
        np.array(y_set.sort())
        x1, y1, x2, y2 = x_set[0], y_set[0], x_set[-1], y_set[-1]

        one_area = image_gray[y1:y2,x1:x2]  # 获取出来符合条件的区域
        all_pixel = []
        for ii in range(one_area.shape[0]):
            for jj in range(one_area.shape[1]):
                if one_area[ii][jj] == 0:
                    all_pixel.append(0)
                else:
                    all_pixel.append(1)
        all_pixel = np.mean(np.array(all_pixel))
        if all_pixel < 0.8:  # 设定阈值 0.8保证识别出来的二维码区域不被包含
            pass
        else:
            img = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.imwrite(save_path+'/{}_{}.jpg'.format(img_path.split('/')[-1].split('.')[0],cc), img)

# img_path='/Users/wywy/Desktop/立学作业纸 正面.jpg'
# save_path='/Users/wywy/Desktop'
# set_pasre(img_path,save_path)


def rotate_bound_white_bg(image, angle):
    '''
    图像旋转背景以白色部分填充
    :param image:
    :param angle:  旋转角度
    :return:
    '''
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # -angle位置参数为角度参数负值表示顺时针旋转; 1.0位置参数scale是调整尺寸比例（图像缩放参数），建议0.75
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 0.75)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    # borderValue 缺失背景填充色彩，此处为白色，可自定义
    return cv2.warpAffine(image, M, (nW, nH), borderValue=(255, 255, 255))
    # borderValue 缺失，默认是黑色（0, 0 , 0）

# img_path='/Users/wywy/Desktop/c_img'
# save_path='/Users/wywy/Desktop/chioce_aug/'
#
# x=0
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         angle=np.random.randint(30)
#         img = cv2.imread(img_path + '/' + file)
#         index=random.sample([-1,1],1)
#         # print(file)
#         name=file.split('.')[0].split('_')[-1]
#         imgRotation = rotate_bound_white_bg(img, angle*int(index[0]))
#         h,w,c=np.shape(imgRotation)
#         # h_setoff,w_setoff=round( h/240,1),round( w/340,1)
#         # print(h_setoff,w_setoff)
#         # a=scipy.misc.imresize(imgRotation,(240,340))
#         # print('/Users/wywy/Desktop/rotate/{}_{}_{}_.jpg'.format(x,h_setoff,w_setoff))
#         cv2.imwrite(save_path+'aug{}_{}.jpg'.format(x,name), imgRotation)
#         x+=1
# print(x)
#

# level_path='/Users/wywy/Desktop/level.txt'
# label_path='/Users/wywy/Desktop/level_label.txt'
# save_path='/Users/wywy/Desktop/level_data'
# def get_label(label_path):
#     list_info=list('ABCDX')
#     with open(label_path, encoding='utf8') as f:
#         lines=f.readlines()
#
#         all_info=[]
#         for l in lines:
#             a=l.strip()
#             if a in list_info:
#                 all_info.append(a)
#
#         return all_info
#
# all_label=get_label(label_path)
# c=0
# with open(level_path) as f:
#     lines=f.readlines()
#     all_URL=[]
#     for line in lines:
#         if line.rsplit()[0]=='#':
#             pass
#         else:
#             all_URL.append(line.rsplit()[0])
#     for index in range(len(all_URL)):
#         urllib.request.urlretrieve(all_URL[index], save_path+'/'+str(c)+'_'+all_label[index]+'.jpg')
#         c+=1










