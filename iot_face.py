import datetime
import os
import cv2
import numpy as np
import pygame
from keras.models import load_model
import time
import iot_camera_dependency as camera

# 初始化 Pygame 音樂模塊
pygame.mixer.init(buffer = 1024)

# 載入模型
photo_cnn_model = load_model('ah.h5')

def photo_predict(image, model):
    # 調整圖像大小
    resized_image = cv2.resize(image, (200, 200))

    # 轉換為灰階
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
    # 正規化圖像
    img = gray_image / 255.0

    if img.shape != (200, 200):
        raise ValueError(f"Image shape {img.shape} is not compatible with the target shape (200, 200)")

    # 重塑圖像以符合模型的輸入要求
    reshaped_img = np.reshape(img, (1, 200, 200, 1))

    # 預測
    predict = np.argmax(model.predict(reshaped_img), axis=1)[0]

    # 類別名稱
    class_name = [ 'sad','happy']
    
    # 返回預測的情緒
    return class_name[predict]

def play_music(emotion):
    if emotion == 'sad':
        pygame.mixer.music.load('sadd.mp3')
    elif emotion == 'happy':
        pygame.mixer.music.load('Happyyyyy.mp3')
    else:
        return  # 如果不是 'sad' 或 'happy'，則不播放音樂
    pygame.mixer.music.play()

    # 使程序保持运行状态，直到音乐播放完毕
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

# 指定存儲照片的目錄
save_dir = 'predicted_images_byInputPhoto'
os.makedirs(save_dir, exist_ok=True)

while True:
    try:
        # 拍攝照片
        img = camera.take_image(num=1, interval=0.5, isFlipV=False, isGray=False, pooling=0, size_scaling=0.061728395)

        # 寫入照片
        # cv2.imwrite(os.path.join(save_dir, f'webcam_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.jpg'), img)
        cv2.imwrite(os.path.join(save_dir, 'aaa.jpg'), img)

        # 預測照片中的情緒
        predicted = photo_predict(img, photo_cnn_model)

        # 顯示預測結果w
        print(f'({datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")}) 檢測到 "{predicted}".')

        emotion_label = np.argmax(predicted)

        # 類別名稱
        class_name = ['sad' ,'happy']
        predicted_emotion = class_name[emotion_label]

        # 根據情緒播放音樂
        play_music(predicted_emotion)
        time.sleep(0.5)

    except Exception as e:
        print(e)


