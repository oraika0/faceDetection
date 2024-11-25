from os.path import exists
from time import sleep
import cv2

CAM_INDEX = 0
DEFAULT_INTERVAL = 0.5
DEFAULT_SIZE = [2592, 1944]


def get_unique_filename(dir: str, file_ext: str):

    if not dir.endswith('/'):
        dir += '/'

    index = 1
    filename = str(index) + file_ext
    
    while exists(dir + filename):
        index += 1
        filename = str(index) + file_ext

    return filename


def take_image(num: int = 1, interval: float = DEFAULT_INTERVAL, isFlipV: bool = False, isGray: bool = False, pooling: int = 0, size_scaling: float = 1, output_dir: str = 'images/') -> int:
    
    num_success = 0
    cap = cv2.VideoCapture(CAM_INDEX)

    if num < 0:
        num = 0

    if interval <= 0:
        interval = DEFAULT_INTERVAL

    if size_scaling <= 0:
        size_scaling = 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_SIZE[0] * size_scaling)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_SIZE[1] * size_scaling)

    if not output_dir.endswith('/'):
        output_dir += '/'

    for i in range(num):
        output_filename = get_unique_filename(output_dir, '.jpg')
            
        isSuccess, frame = cap.read()

        if isSuccess:
            if isFlipV:
                frame = cv2.flip(frame, 0)

            if isGray:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if pooling > 0:
                # Remove even rows and columns from image for *pooling* times.
                for i in range(pooling):
                    frame = frame[::2, ::2]
            
            cv2.imwrite(output_dir + output_filename, frame)
            num_success += 1

            return frame

        else:
            print('Failed to fetch image from /dev/video{} !'.format(CAM_INDEX))

        sleep(interval)

    return num_success
