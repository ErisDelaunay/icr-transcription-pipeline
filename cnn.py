import tensorflow as tf

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # dataset images are 56x56 pixels, and have one color channel
  input_layer0 = tf.reshape(features["x"], [-1, 56, 56, 1])
  input_layer = tf.placeholder_with_default(
      input=input_layer0,
      shape=(None, 56, 56, 1),
      name="input_layer"
  )

  # Convolutional Layer #1
  # Computes 42 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 56, 56, 1]
  # Output Tensor Shape: [batch_size, 56, 56, 42]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=42,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
      kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 56, 56, 42]
  # Output Tensor Shape: [batch_size, 28, 28, 42]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Apply dropout
  pool1_dropout = tf.layers.dropout(
      inputs=pool1,
      rate=0.5,
      training=mode == tf.estimator.ModeKeys.TRAIN)

  # Convolutional Layer #2
  # Computes 42 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 42]
  # Output Tensor Shape: [batch_size, 28, 28, 42]
  conv2 = tf.layers.conv2d(
      inputs=pool1_dropout,
      filters=42,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
      kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 42]
  # Output Tensor Shape: [batch_size, 14, 14, 42]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Apply dropout
  pool2_dropout = tf.layers.dropout(
      inputs=pool2,
      rate=0.5,
      training=mode == tf.estimator.ModeKeys.TRAIN)

  # Convolutional Layer #3
  # Computes 28 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 42]
  # Output Tensor Shape: [batch_size, 14, 14, 28]
  conv3 = tf.layers.conv2d(
      inputs=pool2_dropout,
      filters=28,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
      kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))

  # Pooling Layer #3
  # Third max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 28]
  # Output Tensor Shape: [batch_size, 7, 7, 28]
  pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

  # Apply dropout
  pool3_dropout = tf.layers.dropout(
      inputs=pool3,
      rate=0.5,
      training=mode == tf.estimator.ModeKeys.TRAIN)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 28]
  # Output Tensor Shape: [batch_size, 7 * 7 * 28]
  pool3_flat = tf.reshape(pool3_dropout, [-1, 7 * 7 * 28])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 28]
  # Output Tensor Shape: [batch_size, 100]
  dense = tf.layers.dense(
      inputs=pool3_flat,
      units=100,
      activation=tf.nn.relu,
      kernel_initializer=tf.contrib.layers.xavier_initializer(),
      kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))

  # Add dropout operation; 0.5 probability that element will be kept
  dense_dropout = tf.layers.dropout(
      inputs=dense,
      rate=0.5,
      training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 100]
  # Output Tensor Shape: [batch_size, 23]
  logits = tf.layers.dense(
      inputs=dense_dropout,
      units=33,
      kernel_initializer=tf.contrib.layers.xavier_initializer(),
      kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)