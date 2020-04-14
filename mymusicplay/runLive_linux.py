import tensorflow as tf
import parameters as par
import cv2
import numpy as np
from PIL import ImageOps, Image
import virtkey

v = virtkey.virtkey()

saver = tf.train.import_meta_graph(par.saved_path + str('500.meta'))
sess=tf.Session()
saver.restore(sess, tf.train.latest_checkpoint('./Saved/'))

# Get Operations to restore
graph = sess.graph
# Get Input Graph
X = graph.get_tensor_by_name('Input:0')
    #Y = graph.get_tensor_by_name('Target:0')
    # keep_prob = tf.placeholder(tf.float32)
keep_prob = graph.get_tensor_by_name('Placeholder:0')

    # Get Ops
prediction = graph.get_tensor_by_name('prediction:0')
logits = graph.get_tensor_by_name('logits:0')
accuracy = graph.get_tensor_by_name('accuracy:0')

    # Get the image
cap = cv2.VideoCapture(0)
cv2.namedWindow("camera",1)

lastlaststatus = -1
laststatus = -1

while 1:
    ret, img = cap.read()
    cv2.rectangle(img, (300, 300), (100, 100), (0, 255, 0), 0)
    cv2.imshow("camera",img)
    crop_img = img[100:300, 100:300]
    ycrcb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2YCrCb) # 分解为YUV图像,得到CR分量
    (_, cr, _) = cv2.split(ycrcb)
    cr1 = cv2.GaussianBlur(cr, (35, 35), 0) # 高斯滤波
    _, thresh1 = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # OTSU图像二值化
    cv2.imshow("image1", thresh1)
    thresh1 = (thresh1 * 1.0) / 255
    thresh1 = Image.fromarray(thresh1)
    thresh1 = ImageOps.fit(thresh1, [par.image_size, par.image_size])
    if par.threshold:
        testImage = np.reshape(thresh1, [-1, par.image_size, par.image_size, 1])
    else:
        testImage = np.reshape(thresh1, [-1, par.image_size, par.image_size, 3])
    testImage = testImage.astype(np.float32)
    testY = sess.run(prediction, feed_dict={X: testImage, keep_prob: 1.0})
    if testY[0][1]==1:
        if laststatus != 1 & lastlaststatus != 1:
            print('暂停')
            v.press_keysym(65507) #Ctrl键位
            v.press_keysym(65513) #Alt键位
            v.press_unicode(ord('p')) #模拟字母p
            v.release_unicode(ord('p'))
            v.release_keysym(65513)
            v.release_keysym(65507)
        lastlaststatus = laststatus
        laststatus = 1
    elif testY[0][3]==1:
        print('音量变大')
        v.press_keysym(65507) #Ctrl键位
        v.press_keysym(65513) #Alt键位
        v.press_keysym(65431)
        v.release_keysym(65431)
        v.release_keysym(65513)
        v.release_keysym(65507)
        lastlaststatus = laststatus
        laststatus = 2
    elif testY[0][4]==1:
        if laststatus != 3 & lastlaststatus != 3:
            print('向右')
            v.press_keysym(65507) #Ctrl键位
            v.press_keysym(65513) #Alt键位
            v.press_keysym(65432)
            v.release_keysym(65432)
            v.release_keysym(65513)
            v.release_keysym(65507)
        lastlaststatus = laststatus
        laststatus = 3
    elif testY[0][6]==1:
        if laststatus != 4 & lastlaststatus != 4:
            print('向左')
            v.press_keysym(65507) #Ctrl键位
            v.press_keysym(65513) #Alt键位
            v.press_keysym(65430)
            v.release_keysym(65430)
            v.release_keysym(65513)
            v.release_keysym(65507)
        lastlaststatus = laststatus
        laststatus = 4
    elif testY[0][7]==1:
        print('音量变小')
        v.press_keysym(65507) #Ctrl键位
        v.press_keysym(65513) #Alt键位
        v.press_keysym(65433)
        v.release_keysym(65433)
        v.release_keysym(65513)
        v.release_keysym(65507)
        lastlaststatus = laststatus
        laststatus = 5
    else:
        lastlaststatus = laststatus
        laststatus = 0   
    key1 = cv2.waitKey(30) & 0xff
    if key1 == 27:
        break
cap.release()
cv2.destroyAllWindows()
    
