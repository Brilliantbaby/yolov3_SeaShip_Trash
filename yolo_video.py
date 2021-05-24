import argparse
from yolo import YOLO, detect_video
from PIL import Image
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def detect_img(yolo,path):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image,path)
            r_image.show()
            r_image.save('result.jpg')
    yolo.close_session()

FLAGS = None

if __name__ == '__main__':
    # 添加参数

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
# model的路径
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )
# anchor文件的路径
    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )
# class 文件的路径
    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )
# 如果输入了image参数，其余参数将被忽略
    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
# 视频的输入路径
    parser.add_argument(
        "--input", nargs='?', type=str, required=False, default='SeaShip/val/val.mp4',
        help = "Video input path"
    )
# 视频的输出路径
    parser.add_argument(
        "--output", nargs='?', type=str, default="result_video/ship_result.mp4",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()
    print(FLAGS)
    # 创建一个临时的yolo对象 变量是vars里面的参数值
    detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)

    # if FLAGS.image:
    #     """
    #     Image detection mode, disregard any remaining command line arguments
    #     """
    #     print("Image detection mode")
    #     if "input" in FLAGS:
    #         print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
    #     YOLO.detect_image(YOLO(**vars(FLAGS)))
    # elif "input" in FLAGS:
    #     detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    # else:
    #     print("Must specify at least video_input_path.  See usage with --help.")

