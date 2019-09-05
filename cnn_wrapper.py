import tensorflow as tf
from tensorflow.python import keras


def custom_loss(y_true, y_pred):
    # Compute the binary loss (is char / not char)
    loss_nochar = keras.losses.binary_crossentropy(y_true[:, 0:1], y_pred[:, 0:1], from_logits=True)

    # These are the locations of chars inside the current batch
    idx_chars = tf.where(1 - y_true[:, 0])[:, 0]

    # Compute the cross-entropy loss only for chars
    loss_chars = keras.losses.categorical_crossentropy(
            tf.gather(y_true[:, 1:], idx_chars),
            tf.gather(y_pred[:, 1:], idx_chars), from_logits=True)

    # Sum the two losses (weighted)
    return tf.reduce_sum(loss_nochar) + tf.reduce_sum(loss_chars)*5


def char_accuracy(y_true, y_pred):
    # Returns the accuracy of recognition for the characters
    idx_chars = tf.where(1 - y_true[:, 0])[:, 0]
    return keras.metrics.categorical_accuracy(
            tf.gather(y_true[:, 1:], idx_chars),
            tf.gather(y_pred[:, 1:], idx_chars))


def nochar_accuracy(y_true, y_pred):
    # Returns the accuracy of recognizing chars vs. no-chars
    return keras.metrics.binary_accuracy(y_true[:, 0:1], y_pred[:, 0:1])


class CNN:
    def __init__(self, model_path):
        self.classifier = keras.models.load_model(
            model_path,
            custom_objects={
                'custom_loss':custom_loss,
                'char_accuracy':char_accuracy,
                'nochar_accuracy':nochar_accuracy
            }
        )

    def predict(self, X_test):
        logit_preds = self.classifier.predict(X_test)
        char_preds = tf.nn.softmax(logit_preds[:, 1:], axis=1).numpy()
        notchar_preds = tf.nn.sigmoid(logit_preds[:, :1]).numpy()
        return char_preds, notchar_preds

