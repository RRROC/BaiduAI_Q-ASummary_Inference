import tensorflow as tf

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')


# 定义损失函数
def loss_function(real, pred, pad_index):
    mask = tf.math.logical_not(tf.math.equal(real, pad_index))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)  # 转换为和loss_类型相同的张量
    loss_ *= mask
    return tf.reduce_mean(loss_)
