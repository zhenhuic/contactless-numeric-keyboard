import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def trace_fingertip(img: np.ndarray, ratio: float = 0.05, tip_pixel_num: int = 200) -> (float, float):
    """
    识别光幕上指尖位置坐标
    :param img: np.ndarray，二值化的图片，背景的像素值为 255，手指的像素值为 0
    :param ratio: float，图像中黑色像素点个数占总像素点的比率，当大于这个比率就算图像中有手指，否则算没有
    :param tip_pixel_num: int，取指尖最上端的像素点个数，这些像素点坐标的平均值即代表最终识别的指尖的坐标
    :return: tuple，返回指尖的坐标占与图片宽高的比值（相对坐标）
    """
    mask = (img < 127)  # 黑色像素点的 mask
    black_pixel = img[mask]  # 所以黑色像素点
    black_num = black_pixel.shape[0]  # 黑色像素点的个数
    total_num = np.size(img)  # 总像素点个数

    cnt_pixel = 0
    row_start_idx = 0  # 黑色像素点开始的行索引值
    row_end_idx = 0  # 黑色像素点结束的行索引值（取的黑色像素点个数至少为 tip_pixel_num）

    if black_num > 0 and total_num / black_num >= ratio:
        enough_black_pixel = False
        for i in range(img.shape[0] - 1):
            row = img[i, :]
            cnt = np.size(row[(row < 127)])
            if cnt <= 0:
                continue
            if row_start_idx == 0:
                row_start_idx = i
            cnt_pixel += np.size(row[(row < 127)])
            if cnt_pixel > tip_pixel_num:
                row_end_idx = i
                enough_black_pixel = True
                break
        if not enough_black_pixel:
            row_end_idx = img.shape[0] - 1

        # 返回掩模行号 row_start_idx 至 row_end_idx 所有黑色像素点坐标
        coords = np.argwhere(mask[row_start_idx:row_end_idx, :])
        if coords.shape[0] != 0 or coords.shape[1] != 0:
            tip = np.mean(coords, 0, dtype=np.int)  # 计算坐标的平均值
            return tip[1] / img.shape[1], (tip[0] + row_start_idx) / img.shape[0]  # (x, y)

    return -1, -1


def darken_key_area_color(keypad_img: np.ndarray, key_area: tuple, rate: float = 0.4) -> np.ndarray:
    """
    使某个按键区域变暗，以模拟按键被按下的效果
    :param keypad_img:
    :param key_area:
    :param rate:
    :return:
    """
    key_area_array = keypad_img[key_area[1]:key_area[3], key_area[0]:key_area[2]]
    zero_array = np.zeros_like(key_area_array)
    dim = cv2.addWeighted(key_area_array, (1 - rate), zero_array, rate, 0)
    keypad_img[key_area[1]:key_area[3], key_area[0]:key_area[2]] = dim
    return keypad_img


def draw_chinese_words(img_array, contents, coord, color=(255, 255, 255), size=40):
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_array)

    # PIL图片上打印汉字
    draw = ImageDraw.Draw(img)  # 图片上打印
    font = ImageFont.truetype("simhei.ttf", size, encoding="utf-8")
    draw.text(coord, contents, color, font=font)

    # PIL 图片转 cv2 图片
    img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img_array


def main():
    # img_path = 'image/1.jpg'
    # orig = cv2.imread(img_path)
    # gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

    keypad_value = ((7, 8, 9),
                    (4, 5, 6),
                    (1, 2, 3),
                    (-1, 0, 10))
    keypad_area = (
        ((3, 146, 414, 332), (418, 146, 829, 332), (833, 146, 1244, 332)),
        ((3, 336, 414, 523), (418, 336, 829, 523), (833, 336, 1244, 523)),
        ((3, 527, 414, 714), (418, 527, 829, 714), (833, 527, 1244, 714)),
        ((3, 718, 414, 905), (418, 718, 829, 905), (833, 718, 829, 905)),
    )  # (left, top, right, bottom)

    pointer_color_flag = 0
    key_row_index, key_column_index = -1, -1
    target_floor = None

    keypad_img_path = 'image/keypad.png'
    keypad_img = cv2.imread(keypad_img_path)
    if keypad_img is None:
        print("keypad image lost!")

    screen_brightness = 0  # 光幕的平均亮度
    screen_area = ((150, 70), (490, 410))  # (左上角坐标，右下角坐标)
    calibrated = False  # 是否校准光幕区域的标志

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('未识别到摄像头')
        return
    prev_key = -2
    start_time = 0.0
    key_record = []
    prev_input_key = -2
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if calibrated:
            # 光幕区域校准后执行以下程序
            # 根据计算的光幕亮度进行图像二值化
            ret, binary = cv2.threshold(gray, screen_brightness - 30, 255, cv2.THRESH_BINARY)
            binary = binary[screen_area[0][1]:screen_area[1][1], screen_area[0][0]:screen_area[1][0]]
            # print(np.unique(binary))
            tip_coord = trace_fingertip(binary)  # 识别指尖位置
            show_img = np.copy(keypad_img)

            if 0 < tip_coord[0] < 1:
                coord = (int(show_img.shape[1] * tip_coord[0]), int((show_img.shape[0] - 140) * tip_coord[1]) + 140)
                cv2.circle(show_img, coord, 12, (127, 127, 255), 30)  # 画出指尖位置圆点
                if pointer_color_flag > 0:
                    show_img = darken_key_area_color(show_img, keypad_area[key_row_index][key_column_index])
                    pointer_color_flag -= 1
                # print('fingertip: ', coord)

                x, y = coord
                row, column = -1, -1
                # 键盘按键各区域坐标
                if 140 <= y < 333:
                    row = 0
                elif 333 <= y < 525:
                    row = 1
                elif 525 <= y < 716:
                    row = 2
                elif 716 <= y <= 910:
                    row = 3

                if 0 <= x < 416:
                    column = 0
                elif 416 < x <= 831:
                    column = 1
                elif 831 < x <= 1250:
                    column = 2

                key = keypad_value[row][column]
                # print(key)
                if key == prev_key:
                    curr_time = time.time()
                    if curr_time - start_time > 1:
                        if 0 <= key <= 9:
                            if prev_input_key != key:
                                key_record.append(key)
                                prev_input_key = key
                                print(key)
                                pointer_color_flag = 4

                        elif key == 10:
                            prev_input_key = 10
                            if len(key_record) > 0:
                                floor_num = ''
                                for k in key_record:
                                    floor_num += str(k)
                                print("输入的楼层数：", floor_num)
                                target_floor = floor_num
                                pointer_color_flag = 4
                                key_record.clear()
                            else:
                                if prev_input_key != 10:
                                    print("请先输入楼层数！")
                                    pointer_color_flag = 4
                        elif key == -1:
                            if prev_input_key != -1:
                                key_record.clear()
                                prev_input_key = key
                                print("清空上述输入")
                                pointer_color_flag = 4
                        key_row_index, key_column_index = row, column
                else:
                    prev_input_key = -2
                    prev_key = key
                    start_time = time.time()

            else:
                prev_input_key = -2
                prev_key = -2
                start_time = 0.0

            key_record_num = ''.join([str(x) for x in key_record])
            # for k in key_record:
            #     key_record_num += str(k)
            cv2.putText(show_img, key_record_num, (280, 65), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 128, 0), 6)
            if target_floor is not None:
                cv2.putText(show_img, target_floor, (875, 110), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 128, 0), 6)
            cv2.namedWindow('fingertip', cv2.WINDOW_NORMAL)
            cv2.imshow('fingertip', show_img)
            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            # 在完成光幕区域校准之前执行以下程序
            # 校准光幕放置区域
            img = cv2.rectangle(frame, screen_area[0], screen_area[1], (200, 200, 0), 2)
            cv2.namedWindow('calibrate', cv2.WINDOW_NORMAL)
            cv2.imshow('calibrate', img)
            if cv2.waitKey(1) & 0xFF == ord('c'):
                # 取中心 40*40 的像素样本计算光幕平均亮度
                sample = gray[220:260, 300:340]
                screen_brightness = int(np.mean(sample))  # 给光幕亮度变量赋值
                print('screen brightness: ', screen_brightness)
                calibrated = True  # 设置校准标志为已校准完成
                cv2.destroyWindow('calibrate')
    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    main()
