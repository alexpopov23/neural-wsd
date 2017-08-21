import tensorflow as tf


with tf.Session(graph=tf.Graph()) as session:
    saver = tf.train.Saver()
    saver.restore(session, ос.path.joint("save-path", "model.ckpt-25000.data-00000-of-00001"))
