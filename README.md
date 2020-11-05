### 本demo是基于![Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)的Python MNN版本进行的修改，实现了从人脸抓取到人脸比对的整个流程
###### 详细的网络模型介绍和实现方式可参考我的![CSDN博客](https://blog.csdn.net/LQ_qing/article/details/109238853)。在该博客中，人脸抓取使用的是MTCNN，但MTCNN和Facenet合在一起后，完成一次抓取和识别的时间花费接近1s，所以我才把MTCNN换成了上述网络模型
### 基础功能
###### 1，实时读取网络摄像头的rtsp流数据
###### 2，利用Python多进程和多线程的方式，并行处理8路摄像头数据，保证数据的实时性
###### 3，比对完人脸数据后，进行结果的保存
### 未完待续。。。。。
