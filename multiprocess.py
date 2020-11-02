import time
import multiprocessing as mp
import cv2

def image_put(q, user, pwd, ip, channel=1):
    cap = cv2.VideoCapture("rtsp://%s:%s@%s//Streaming/Channels/%d" % (user, pwd, ip, channel))
    if cap.isOpened():
        print('HIKVISION')
    else:
        cap = cv2.VideoCapture("rtsp://%s:%s@%s/cam/realmonitor?channel=%d&subtype=0" % (user, pwd, ip, channel))
        print('DaHua')

    while True:
        q.put(cap.read()[1])
        q.get() if q.qsize() > 1 else time.sleep(0.01)


def image_get(q, window_name):
    cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    while True:
        frame = q.get()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow(window_name, image)
        cv2.waitKey(1)

def run_multi_camera():
    user_name, user_pwd = "admin", "flm2019hb"
    camera_ip_l = [
        "192.168.101.107",  # ipv4
        "192.168.101.107",
        "192.168.101.107",
        "192.168.101.107"
    ]

    mp.set_start_method(method='spawn')  # init
    queues = [mp.Queue(maxsize=4) for _ in camera_ip_l]

    processes = []
    for queue, camera_ip in zip(queues, camera_ip_l):
        processes.append(mp.Process(target=image_put, args=(queue, user_name, user_pwd, camera_ip)))
        processes.append(mp.Process(target=image_get, args=(queue, camera_ip)))

    for process in processes:
        process.daemon = True
        process.start()
    for process in processes:
        process.join()


def image_collect(queue_list, camera_ip_l):
    import numpy as np

    """show in single opencv-imshow window"""
    window_name = "%s_and_so_no" % camera_ip_l[0]
    cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    while True:
        imgs = [q.get() for q in queue_list]
        imgs = np.concatenate(imgs, axis=1)
        cv2.imshow(window_name, imgs)
        cv2.waitKey(1)

    # """show in multiple opencv-imshow windows"""
    # [cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    #  for window_name in camera_ip_l]
    # while True:
    #     for window_name, q in zip(camera_ip_l, queue_list):
    #         cv2.imshow(window_name, q.get())
    #         cv2.waitKey(1)


def run_multi_camera_in_a_window():
    user_name, user_pwd = "admin", "flm2019hb"
    camera_ip_l = [
        "192.168.101.107",  # ipv4
        "192.168.101.107",
        "192.168.101.107",
        "192.168.101.107"
    ]

    mp.set_start_method(method='spawn')  # init
    queues = [mp.Queue(maxsize=4) for _ in camera_ip_l]

    processes = [mp.Process(target=image_collect, args=(queues, camera_ip_l))]
    for queue, camera_ip in zip(queues, camera_ip_l):
        processes.append(mp.Process(target=image_put, args=(queue, user_name, user_pwd, camera_ip)))

    for process in processes:
        process.daemon = True  # setattr(process, 'deamon', True)
        process.start()
    for process in processes:
        process.join()


def run_single_camera():
    user_name, user_pwd, camera_ip = "admin", "flm2019hb", "192.168.101.107"

    mp.set_start_method(method='spawn')  # init
    queue = mp.Queue(maxsize=2)
    processes = [mp.Process(target=image_put, args=(queue, user_name, user_pwd, camera_ip)),
                 mp.Process(target=image_get, args=(queue, camera_ip))]

    [process.start() for process in processes]
    [process.join() for process in processes]

def run():
    # run_opencv_camera()  # slow, with only 1 thread
    run_single_camera()  # quick, with 2 threads
    # run_multi_camera() # with 1 + n threads
    # run_multi_camera_in_a_window()  # with 1 + n threads


if __name__ == '__main__':
    run()