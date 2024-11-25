import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model

# 設置數據路徑
base_dir = '/data/s10959024/pose/iot/Train/'
val_dir = base_dir  # 使用訓練集作為驗證集
categories = ['sad', 'happy']

# 設置模型參數
img_width, img_height = 200, 200
batch_size = 32
epochs = 300

# 圖像預處理 - 數據生成器，設為灰階
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# 從目錄加載訓練和驗證數據
train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale'  # 設為灰階
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale',  # 設為灰階
    shuffle=False
)

# 建立模型，設定第一層為灰階輸入
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(categories), activation='softmax')
])

# 繪製模型結構
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# 編譯模型
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# 設置 ModelCheckpoint 回調來保存最佳模型
best_model_path = '/data/s10959024/pose/iot/result/best_model/best_model.h5'
checkpoint = ModelCheckpoint(best_model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)

# 初始化歷史記錄
all_history = {
    'accuracy': [],
    'val_accuracy': [],
    'loss': [],
    'val_loss': []
}

# 訓練模型
for epoch in range(epochs):
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=1,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        callbacks=[checkpoint]
    )

    # 累積歷史數據
    all_history['accuracy'].extend(history.history['accuracy'])
    all_history['val_accuracy'].extend(history.history['val_accuracy'])
    all_history['loss'].extend(history.history['loss'])
    all_history['val_loss'].extend(history.history['val_loss'])

# 保存最後的模型
model.save('/data/s10959024/pose/iot/result/final_model.h5')

# 訓練完成後繪製圖表
save_path = '/data/s10959024/pose/iot/result/training_results'
os.makedirs(save_path, exist_ok=True)

plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
plt.plot(all_history['accuracy'], label='Training Accuracy')
plt.plot(all_history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(all_history['loss'], label='Training Loss')
plt.plot(all_history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig(f'{save_path}/training_validation_process.png')
plt.close()




