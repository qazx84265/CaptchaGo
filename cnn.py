# -*- coding: UTF-8 -*-

from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import random

# 验证码字符集
import string
characters = string.digits + string.ascii_uppercase
#print(characters)

width, height, n_len, n_class = 170, 80, 4, len(characters)


# 数据生成器
from keras.utils.np_utils import to_categorical

def gen(batch_size=32):
    X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]
    generator = ImageCaptcha(width=width, height=height)
    while True:
        for i in range(batch_size):
            random_str = ''.join([random.choice(characters) for j in range(4)])
            X[i] = generator.generate_image(random_str)
            for j, ch in enumerate(random_str):
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1
        yield X, y


# 测试数据生成器
def decode(y):
     y = np.argmax(np.array(y), axis=2)[:,0]
     return ''.join([characters[x] for x in y])

# 使用生成器
# fig = plt.figure()
#X, y = next(gen(1))
#plt.imshow(X[0])
#plt.title(decode(y))
## show image
#plt.show()
#
# # save image
# # fig.savefig('temp.png')


# 定义网络结构
from keras.models import *
from keras.layers import *
from keras import optimizers
#from keras import losses
from keras import metrics

input_tensor = Input((height, width, 3))
x = input_tensor
for i in range(4):
    x = Convolution2D(32*2**i, 3, 3, activation='relu')(x)
    x = Convolution2D(32*2**i, 3, 3, activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dropout(0.25)(x)
x = [Dense(n_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(4)]
model = Model(input=input_tensor, output=x)

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])


# 神经网络模型可视化
# from keras.utils.visualize_util import plot
# from IPython.display import Image
#
# plot(model, to_file="model.png", show_shapes=True)
# Image('model.png')


# 模型训练
model.fit_generator(gen(),
                    samples_per_epoch=51200,
                    nb_epoch=5,
                    nb_worker=2,
                    pickle_safe=True,
                    validation_data=gen(),
                    nb_val_samples=1280)


# 测试模型
#from keras.models import load_model
#model=load_model('model.h5')
#
#X, y = next(gen(1))
#y_pred = model.predict(X)
#
##fig = plt.figure()
#plt.title('real: %s\npred:%s'%(decode(y), decode(y_pred)))
#plt.imshow(X[0], cmap='gray')
#plt.show()

#fig.savefig('test.png')


# 计算模型准确率
#from tqdm import tqdm
#def evaluate(model, batch_num=20):
#    batch_acc = 0
#    generator = gen()
#    for i in tqdm(range(batch_num)):
#        X, y = next(generator)
#        y_pred = model.predict(X)
#        y_pred = np.argmax(y_pred, axis=2).T
#        y_true = np.argmax(y, axis=2).T
#        batch_acc += np.mean(map(np.array_equal, y_true, y_pred))
#    return batch_acc / batch_num
#
#evaluate(model)


# 保持模型
model.save('cnn.h5')


# 转换为CoreML模型
import coremltools
coreml_model = coremltools.converters.keras.convert(model, input_names=['captchaImage'], image_input_names='captchaImage')
#coreml_model = coremltools.converters.keras.convert(model)
coreml_model.author = 'FB'
#coreml_model.license = 'MIT'
coreml_model.short_description = 'cracking captcha with digits or letters'
#coreml_model.input_description['captchaImage'] = 'captcha image'
#coreml_model.output_description['captchaCode'] = 'captcha code recgnized'
coreml_model.save('Captcha.mlmodel')

