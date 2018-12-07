import cv2
import numpy as np

cv2.namedWindow("test")  # 打卡一个窗口 标题为 test
cap = cv2.VideoCapture(0)  # 打开笔记本的内置摄像头，0为计算机默认的摄像头
# cap = cv2.VideoCapture("test.mp4")  #
success, frame = cap.read()  #返回true/Flase,和当前截取的这一帧的图片,三维矩阵，(成功截取返回ture)

# classifier = cv2.CascadeClassifier("")  # 确保此 xml 文件与该 py文件在一个文件夹下 否则需要修改绝对路径

# haarcascade_frontalface_default.xml
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")  # 产生一个检测器,检测的依据全都储存在参数所代表的那个xml文件中

while success:
    success, frame = cap.read()  # 再次读取视频文件
    size = frame.shape[:2]  # 图片是三维矩阵，要获取图片的长和宽，必须的知道图片的shape，一张图片的shape为【H，W，C】，取前两个即得到图片的宽高
    image = np.zeros(size, dtype=np.float16)  # 生成了一个 类型为 float16的 0 矩阵 size和获取出来的图片一样
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 对读取出来的图片进行颜色空间转换，转换成灰度图
    cv2.equalizeHist(image, image)  #灰度图片直方图均衡化（只接受灰度图，这就是前面为啥要转成灰度图的原因了）
    divisor = 8  # 限制得到目标区域大小和原始图片大小的比例，即最小的检测框为整张图片的八分之一大
    h, w = size  #获取每一帧图片的高/宽
    minSize = (w // divisor, h // divisor)  # 最小的检测框为整张图片的八分之一大
    faceRects = classifier.detectMultiScale(image, 1.2, 2, cv2.CASCADE_SCALE_IMAGE, minSize)

    #这是一个人脸检测的函数，image为待检测图片，一般为灰度图像加快检测速度，1.2表示在前后两次相继的扫描中，搜索窗口的比例系数。默认为1.1即每次搜索窗口依次扩大10%.
    #2表示构成检测目标的相邻矩形的最小个数(默认为3个)。。 这些都是官方给出的。这些参数自己调整以达到自己想要的效果
    #返回值为框住的人脸的坐标（左上角X坐标，左上角Y坐标，框的宽，高）
    if len(faceRects) > 0:
        for faceRect in faceRects:  # 遍历所得的所有人脸的坐标
            x, y, w, h = faceRect  # 返回左上角X坐标，左上角Y坐标，框的宽，高
            cv2.rectangle(frame, (x, y), (x + h, y + w), (0, 255, 0), 2)  # 在未做灰度处理前的图片上画矩形框，frame为图片，(x, y)为左上角X，Y坐标，
            #(x + h, y + w)为右下角XY坐标，(0, 255, 0)表示红色，2表示画矩形的线宽为2

            # 这里的眼睛嘴巴的坐标是根据人脸框来确定的，所以这里算了下眼睛和嘴巴的大致位子。位置的计算方式不必过分深究。
            # cv2.circle(frame, (x + w // 4, y + h // 4 + 30), min(w // 8, h // 8), (255, 0, 0))  # 左眼  参数1为图片，参数2为圆圈中心点坐标，参数3为半径，参数4为颜色
            # cv2.circle(frame, (x + 3 * w // 4, y + h // 4 + 30), min(w // 8, h // 8), (255, 0, 0))  # 右眼     同上
            # cv2.rectangle(frame, (x + 3 * w // 8, y + 3 * h // 4), (x + 5 * w // 8, y + 7 * h // 8), (255, 0, 0))  # 嘴巴 参数1图片，参数2左上角坐标，参数3右下脚坐标，参数4颜色
    cv2.imshow("test", frame)  # 把每一帧的图show 出来
    key = cv2.waitKey(10)  # 不断刷新图像，10ms刷新一下，返回一个数字，数字代表的意思不知道
    c = chr(key & 255)  #  把数字转换成字符串，具体怎么转的不知道，计算机基础没学好
    if c in ['q', 'Q', chr(27)]:  # 如果C在这个列表的三个中就break 具体为啥是这三个字母我也不清楚，有空了可以去研究下给我说，嘿嘿
        break
cv2.destroyWindow('test')  # 关闭所有窗口