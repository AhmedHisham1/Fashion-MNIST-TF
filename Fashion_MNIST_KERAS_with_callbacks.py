import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from time import time
from sklearn.metrics import confusion_matrix


class CustomCallback(tf.keras.callbacks.Callback):
    saved_logs = []
    def on_train_begin(self, logs=None):
        self.saved_logs = []
    def on_epoch_end(self, epoch, logs={}):
        # print('\nlogs:\n{}'.format(logs))     # For Debugging
        if logs.get('accuracy') > 0.9:
            print("\nReached more than 90 percent accuracy. Ending the training!")
            self.model.stop_training = True
        
    def on_batch_end(self, batch, logs={}):
        self.saved_logs.append(logs)


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

print("x_train shape: {}".format(x_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("x_test shape: {}".format(x_test.shape))
print("y_test shape: {}".format(y_test.shape))

x_train, x_test = x_train / 255.0, x_test / 255.0       # Features Normalization

plt.imshow(x_train[768,:,:], cmap='Greys')

print(type(x_train))
print(x_train.shape)
print(y_train.shape)

#reshaping to add the channel dim
x_train = x_train.reshape((x_train.shape[0],x_train.shape[1], x_train.shape[2], 1))
# y_train = y_train.reshape((y_train.shape[0],y_train.shape[1], y_train.shape[2], 1))
x_test = x_test.reshape((x_test.shape[0],x_test.shape[1], x_test.shape[2], 1))
# y_test = y_test.reshape((y_test.shape[0],y_test.shape[1], y_test.shape[2], 1))

print("x_train shape: {}".format(x_train.shape))

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(kernel_size=[5,5], filters=32, activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(kernel_size=[5,5], filters=64, activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

mycallback = CustomCallback()

model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=512, epochs=5, validation_split=0.0, callbacks=[mycallback])
model.evaluate(x_test, y_test)


def print_confusion_matrix(v_xs, v_ys):
    cls_true = v_ys
    cls_pred = np.argmax(model.predict(v_xs), axis=1).T
    cm = confusion_matrix(y_true=cls_true, y_pred=cls_pred)
    
    plt.figure()
    # plt.subplot(121)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.tight_layout()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, range(10))
    plt.yticks(tick_marks, range(10))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title("Confusion Matrix")
    
    # Print the confusion matrix as text.
    # print(cm)


img_no = 1234

plt.figure()
plt.imshow(x_test[img_no,:,:,0], cmap='Greys')
pred = np.argmax(model.predict(x_test, batch_size=None, verbose=1), axis=1)[img_no]
print("prediction for this image: {}".format(pred))
print("Correct classification is: {}".format(y_test[img_no]))

print_confusion_matrix(x_test, y_test.reshape((-1, 1)))


# Plotting acc and loss over epochs

try:
    acc_log = [mycallback.saved_logs[i]['accuracy'] for i in range(len(mycallback.saved_logs))]
    loss_log = [mycallback.saved_logs[i]['loss'] for i in range(len(mycallback.saved_logs))]
    plt.figure()
    plt.subplot(121)
    plt.plot([i for i in range(len(mycallback.saved_logs))], acc_log)
    plt.title('acc')
    plt.xlabel('batches')

    # plt.figure()
    plt.subplot(122)
    plt.plot([i for i in range(len(mycallback.saved_logs))], loss_log)
    plt.title('loss')
    plt.xlabel('batches')
    plt.show()
except:
    print('exception at assigning acc_log and loss_log')

    