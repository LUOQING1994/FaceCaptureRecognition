# -*-coding: utf-8 -*-
"""
    @Project: faceRecognition
    @File   : face_recognition.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2018-12-07 11:33:30
"""
import facenet
import tensorflow as tf
import numpy as np
class facenetEmbedding:
    def __init__(self,model_path):
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        # Load the model
        facenet.load_model(model_path)
        # Get input and output tensors
        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.tf_embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    def  get_embedding(self,images):
        feed_dict = {self.images_placeholder: images, self.phase_train_placeholder: False}
        embedding = self.sess.run(self.tf_embeddings, feed_dict=feed_dict)
        return embedding
    def free(self):
        self.sess.close()