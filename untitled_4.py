# =========================
# 人体跟踪云台 + 手势控制（修复版 v3）
# =========================

from libs.PipeLine import PipeLine, ScopedTiming
from libs.AIBase import AIBase
from libs.AI2D import Ai2d
from media.media import *
from media.sensor import *
import nncase_runtime as nn
import ulab.numpy as np
import aicube
import gc
import time
import utime

from machine import I2C, FPIOA
from servo import Servos


# =========================
# 可调参数区
# =========================
LOST_FRAMES_TO_UNLOCK = 8
MAX_JUMP_PX = 250

EMA_ALPHA_X = 0.6
EMA_ALPHA_Y = 0.4

DEADBAND_X = 8
DEADBAND_Y = 15

SERVO_UPDATE_MS = 25

MAX_STEP_X_DEG = 8.0
MAX_STEP_Y_DEG = 4.0

MIN_STEP_X_DEG = 0.3
MIN_STEP_Y_DEG = 0.5

X_MIN, X_MAX = 0, 270
Y_MIN, Y_MAX = 0, 90

Y_OUT_EMA_ALPHA = 0.4
Y_SOFT_ZONE_DEG = 8.0

GESTURE_START = "thumbUp"
GESTURE_STOP = "fist"
GESTURE_HOLD_FRAMES = 8


# =========================
# I2C & 舵机初始化
# =========================
fpioa = FPIOA()
fpioa.set_function(11, FPIOA.IIC2_SCL)
fpioa.set_function(12, FPIOA.IIC2_SDA)

i2c = I2C(2, freq=100000)

servo_x = Servos(i2c, degrees=270)
servo_y = Servos(i2c, degrees=270)

x_angle = 135.0
y_angle = 45.0

servo_x.position(0, x_angle)
servo_y.position(1, y_angle)


# =========================
# 工具函数
# =========================
def clamp(v, vmin, vmax):
    if v < vmin:
        return vmin
    if v > vmax:
        return vmax
    return v

def ema(prev, cur, alpha):
    if prev is None:
        return cur
    return prev + alpha * (cur - prev)

def dist2(ax, ay, bx, by):
    dx = ax - bx
    dy = ay - by
    return dx * dx + dy * dy

def y_soft_limit_scale(angle):
    margin = min(angle - Y_MIN, Y_MAX - angle)
    if margin >= Y_SOFT_ZONE_DEG:
        return 1.0
    if margin <= 0:
        return 0.0
    return margin / Y_SOFT_ZONE_DEG


# =========================
# PID 控制器
# =========================
class PID:
    def __init__(self, p=0.01, i=0.0, d=0.01):
        self.kp = p
        self.ki = i
        self.kd = d
        self.target = 0
        self.error = 0
        self.last_error = 0
        self.integral = 0

    def update(self, current_value):
        self.error = self.target - current_value
        if abs(self.error) < 5:
            return 0
        self.integral += self.error
        self.integral = clamp(self.integral, -500, 500)
        derivative = self.error - self.last_error
        output = (self.kp * self.error) + (self.ki * self.integral) + (self.kd * derivative)
        self.last_error = self.error
        return output

    def set_target(self, target):
        self.target = target
        self.integral = 0
        self.last_error = 0

    def reset(self):
        self.integral = 0
        self.last_error = 0


x_pid = PID(p=0.025, i=0.0, d=0.002)
y_pid = PID(p=0.018, i=0.0, d=0.002)


# =========================
# 人体检测类
# =========================
class PersonDetectionApp(AIBase):
    def __init__(self, kmodel_path, model_input_size, labels, anchors,
                 confidence_threshold=0.2, nms_threshold=0.5, nms_option=False,
                 strides=[8,16,32], rgb888p_size=[224,224], display_size=[1920,1080], debug_mode=0):
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)
        self.kmodel_path = kmodel_path
        self.model_input_size = model_input_size
        self.labels = labels
        self.anchors = anchors
        self.strides = strides
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.nms_option = nms_option
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0], 16), rgb888p_size[1]]
        self.display_size = [ALIGN_UP(display_size[0], 16), display_size[1]]
        self.debug_mode = debug_mode
        self.ai2d = Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)

    def config_preprocess(self, input_image_size=None):
        with ScopedTiming("set preprocess config", self.debug_mode > 0):
            ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size
            top, bottom, left, right = self.get_padding_param()
            self.ai2d.pad([0, 0, 0, 0, top, bottom, left, right], 0, [0, 0, 0])
            self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
            self.ai2d.build([1, 3, ai2d_input_size[1], ai2d_input_size[0]],
                           [1, 3, self.model_input_size[1], self.model_input_size[0]])

    def postprocess(self, results):
        with ScopedTiming("postprocess", self.debug_mode > 0):
            dets = aicube.anchorbasedet_post_process(results[0], results[1], results[2],
                                                      self.model_input_size, self.rgb888p_size,
                                                      self.strides, len(self.labels),
                                                      self.confidence_threshold, self.nms_threshold,
                                                      self.anchors, self.nms_option)
            return dets

    def draw_result(self, pl, dets, locked_index=-1):
        if dets:
            for i, det_box in enumerate(dets):
                x1, y1, x2, y2 = det_box[2], det_box[3], det_box[4], det_box[5]
                w = int(float(x2 - x1) * self.display_size[0] // self.rgb888p_size[0])
                h = int(float(y2 - y1) * self.display_size[1] // self.rgb888p_size[1])
                x = int(x1 * self.display_size[0] // self.rgb888p_size[0])
                y = int(y1 * self.display_size[1] // self.rgb888p_size[1])

                if i == locked_index:
                    pl.osd_img.draw_rectangle(x, y, w, h, color=(0, 255, 0, 255), thickness=4)
                    pl.osd_img.draw_string_advanced(x, y - 35, 28, "TRACKING", color=(0, 255, 0, 255))
                    cx = x + w // 2
                    cy = y + h // 2
                    pl.osd_img.draw_cross(cx, cy, color=(255, 0, 0, 255), thickness=3)
                else:
                    pl.osd_img.draw_rectangle(x, y, w, h, color=(255, 255, 0, 255), thickness=2)
                    pl.osd_img.draw_string_advanced(x, y - 30, 24, "person", color=(255, 255, 0, 255))

    def get_padding_param(self):
        dst_w = self.model_input_size[0]
        dst_h = self.model_input_size[1]
        input_width = self.rgb888p_size[0]
        input_high = self.rgb888p_size[1]
        ratio_w = dst_w / input_width
        ratio_h = dst_h / input_high
        ratio = min(ratio_w, ratio_h)
        new_w = int(ratio * input_width)
        new_h = int(ratio * input_high)
        dw = (dst_w - new_w) / 2
        dh = (dst_h - new_h) / 2
        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        left = int(round(dw - 0.1))
        right = int(round(dw - 0.1))
        return top, bottom, left, right


# =========================
# 手掌检测类
# =========================
class HandDetApp(AIBase):
    def __init__(self, kmodel_path, labels, model_input_size, anchors,
                 confidence_threshold=0.2, nms_threshold=0.5, nms_option=False,
                 strides=[8,16,32], rgb888p_size=[1920,1080], display_size=[1920,1080], debug_mode=0):
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)
        self.kmodel_path = kmodel_path
        self.labels = labels
        self.model_input_size = model_input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.anchors = anchors
        self.strides = strides
        self.nms_option = nms_option
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0], 16), rgb888p_size[1]]
        self.display_size = [ALIGN_UP(display_size[0], 16), display_size[1]]
        self.debug_mode = debug_mode
        self.ai2d = Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)

    def config_preprocess(self, input_image_size=None):
        with ScopedTiming("set preprocess config", self.debug_mode > 0):
            ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size
            top, bottom, left, right = self.get_padding_param()
            self.ai2d.pad([0, 0, 0, 0, top, bottom, left, right], 0, [114, 114, 114])
            self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
            self.ai2d.build([1, 3, ai2d_input_size[1], ai2d_input_size[0]],
                           [1, 3, self.model_input_size[1], self.model_input_size[0]])

    def postprocess(self, results):
        with ScopedTiming("postprocess", self.debug_mode > 0):
            dets = aicube.anchorbasedet_post_process(results[0], results[1], results[2],
                                                      self.model_input_size, self.rgb888p_size,
                                                      self.strides, len(self.labels),
                                                      self.confidence_threshold, self.nms_threshold,
                                                      self.anchors, self.nms_option)
            return dets

    def get_padding_param(self):
        dst_w = self.model_input_size[0]
        dst_h = self.model_input_size[1]
        input_width = self.rgb888p_size[0]
        input_high = self.rgb888p_size[1]
        ratio_w = dst_w / input_width
        ratio_h = dst_h / input_high
        ratio = min(ratio_w, ratio_h)
        new_w = int(ratio * input_width)
        new_h = int(ratio * input_high)
        dw = (dst_w - new_w) / 2
        dh = (dst_h - new_h) / 2
        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        left = int(round(dw - 0.1))
        right = int(round(dw + 0.1))
        return top, bottom, left, right


# =========================
# 手势关键点分类类
# =========================
class HandKPClassApp(AIBase):
    def __init__(self, kmodel_path, model_input_size, rgb888p_size=[1920,1080], display_size=[1920,1080], debug_mode=0):
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)
        self.kmodel_path = kmodel_path
        self.model_input_size = model_input_size
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0], 16), rgb888p_size[1]]
        self.display_size = [ALIGN_UP(display_size[0], 16), display_size[1]]
        self.crop_params = []
        self.debug_mode = debug_mode
        self.ai2d = Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)

    def config_preprocess(self, det, input_image_size=None):
        with ScopedTiming("set preprocess config", self.debug_mode > 0):
            ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size
            self.crop_params = self.get_crop_param(det)
            self.ai2d.crop(self.crop_params[0], self.crop_params[1], self.crop_params[2], self.crop_params[3])
            self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
            self.ai2d.build([1, 3, ai2d_input_size[1], ai2d_input_size[0]],
                           [1, 3, self.model_input_size[1], self.model_input_size[0]])

    def postprocess(self, results):
        with ScopedTiming("postprocess", self.debug_mode > 0):
            results = results[0].reshape(results[0].shape[0] * results[0].shape[1])
            results_show = np.zeros(results.shape, dtype=np.int16)
            results_show[0::2] = results[0::2] * self.crop_params[3] + self.crop_params[0]
            results_show[1::2] = results[1::2] * self.crop_params[2] + self.crop_params[1]
            gesture = self.hk_gesture(results_show)
            return gesture

    def get_crop_param(self, det_box):
        x1, y1, x2, y2 = det_box[2], det_box[3], det_box[4], det_box[5]
        w, h = int(x2 - x1), int(y2 - y1)
        length = max(w, h) / 2
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        ratio_num = 1.26 * length
        x1_kp = int(max(0, cx - ratio_num))
        y1_kp = int(max(0, cy - ratio_num))
        x2_kp = int(min(self.rgb888p_size[0] - 1, cx + ratio_num))
        y2_kp = int(min(self.rgb888p_size[1] - 1, cy + ratio_num))
        w_kp = int(x2_kp - x1_kp + 1)
        h_kp = int(y2_kp - y1_kp + 1)
        return [x1_kp, y1_kp, w_kp, h_kp]

    def hk_vector_2d_angle(self, v1, v2):
        v1_x, v1_y, v2_x, v2_y = v1[0], v1[1], v2[0], v2[1]
        v1_norm = np.sqrt(v1_x * v1_x + v1_y * v1_y)
        v2_norm = np.sqrt(v2_x * v2_x + v2_y * v2_y)
        dot_product = v1_x * v2_x + v1_y * v2_y
        cos_angle = dot_product / (v1_norm * v2_norm)
        angle = np.acos(cos_angle) * 180 / np.pi
        return angle

    def hk_gesture(self, results):
        angle_list = []
        for i in range(5):
            angle = self.hk_vector_2d_angle(
                [(results[0] - results[i*8+4]), (results[1] - results[i*8+5])],
                [(results[i*8+6] - results[i*8+8]), (results[i*8+7] - results[i*8+9])]
            )
            angle_list.append(angle)

        thr_angle, thr_angle_thumb, thr_angle_s = 65., 53., 49.
        gesture_str = None

        if 65535. not in angle_list:
            if (angle_list[0] > thr_angle_thumb) and (angle_list[1] > thr_angle) and (angle_list[2] > thr_angle) and (angle_list[3] > thr_angle) and (angle_list[4] > thr_angle):
                gesture_str = "fist"
            elif (angle_list[0] < thr_angle_s) and (angle_list[1] < thr_angle_s) and (angle_list[2] < thr_angle_s) and (angle_list[3] < thr_angle_s) and (angle_list[4] < thr_angle_s):
                gesture_str = "five"
            elif (angle_list[0] < thr_angle_s) and (angle_list[1] < thr_angle_s) and (angle_list[2] > thr_angle) and (angle_list[3] > thr_angle) and (angle_list[4] > thr_angle):
                gesture_str = "gun"
            elif (angle_list[0] < thr_angle_s) and (angle_list[1] < thr_angle_s) and (angle_list[2] > thr_angle) and (angle_list[3] > thr_angle) and (angle_list[4] < thr_angle_s):
                gesture_str = "love"
            elif (angle_list[0] > 5) and (angle_list[1] < thr_angle_s) and (angle_list[2] > thr_angle) and (angle_list[3] > thr_angle) and (angle_list[4] > thr_angle):
                gesture_str = "one"
            elif (angle_list[0] < thr_angle_s) and (angle_list[1] > thr_angle) and (angle_list[2] > thr_angle) and (angle_list[3] > thr_angle) and (angle_list[4] < thr_angle_s):
                gesture_str = "six"
            elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] < thr_angle_s) and (angle_list[2] < thr_angle_s) and (angle_list[3] < thr_angle_s) and (angle_list[4] > thr_angle):
                gesture_str = "three"
            elif (angle_list[0] < thr_angle_s) and (angle_list[1] > thr_angle) and (angle_list[2] > thr_angle) and (angle_list[3] > thr_angle) and (angle_list[4] > thr_angle):
                gesture_str = "thumbUp"
            elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] < thr_angle_s) and (angle_list[2] < thr_angle_s) and (angle_list[3] > thr_angle) and (angle_list[4] > thr_angle):
                gesture_str = "yeah"

        return gesture_str


# =========================
# 目标锁定器
# =========================
class TargetLocker:
    def __init__(self, display_size, rgb_size):
        self.display_size = display_size
        self.rgb_size = rgb_size
        self.locked = False
        self.lost = 0
        self.prev_cx = None
        self.prev_cy = None
        self.locked_index = -1

    def reset(self):
        self.locked = False
        self.lost = 0
        self.prev_cx = None
        self.prev_cy = None
        self.locked_index = -1

    def _to_display(self, x1, y1, x2, y2):
        X1 = int(x1 * self.display_size[0] // self.rgb_size[0])
        Y1 = int(y1 * self.display_size[1] // self.rgb_size[1])
        X2 = int(x2 * self.display_size[0] // self.rgb_size[0])
        Y2 = int(y2 * self.display_size[1] // self.rgb_size[1])
        return X1, Y1, X2, Y2

    def update(self, dets):
        if not dets:
            if self.locked:
                self.lost += 1
                if self.lost >= LOST_FRAMES_TO_UNLOCK:
                    self.reset()
            return -1, None

        cand = []
        for i, det in enumerate(dets):
            x1, y1, x2, y2 = det[2], det[3], det[4], det[5]
            X1, Y1, X2, Y2 = self._to_display(x1, y1, x2, y2)
            cx = (X1 + X2) / 2
            cy = (Y1 + Y2) / 2
            area = (X2 - X1) * (Y2 - Y1)
            cand.append((i, cx, cy, area))

        if not self.locked or self.prev_cx is None:
            best = max(cand, key=lambda x: x[3])
            self.locked = True
            self.lost = 0
            self.locked_index = best[0]
            self.prev_cx = best[1]
            self.prev_cy = best[2]
            return best[0], (best[1], best[2])

        best = None
        best_d2 = float('inf')
        for item in cand:
            d2 = dist2(item[1], item[2], self.prev_cx, self.prev_cy)
            if d2 < best_d2:
                best_d2 = d2
                best = item

        if best is None or best_d2 > (MAX_JUMP_PX * MAX_JUMP_PX):
            self.lost += 1
            if self.lost >= LOST_FRAMES_TO_UNLOCK:
                self.reset()
            return -1, None

        self.locked = True
        self.lost = 0
        self.locked_index = best[0]
        self.prev_cx = best[1]
        self.prev_cy = best[2]
        return best[0], (best[1], best[2])


# =========================
# 显示配置
# =========================
display = "lcd3_5"

if display == "hdmi":
    display_mode = "hdmi"
    display_size = [1920, 1080]
    rgb888p_size = [1920, 1080]
elif display == "lcd3_5":
    display_mode = "st7701"
    display_size = [800, 480]
    rgb888p_size = [1920, 1080]
else:
    display_mode = "st7701"
    display_size = [640, 480]
    rgb888p_size = [1280, 960]

# =========================
# 模型路径和参数
# =========================
person_kmodel_path = "/sdcard/examples/kmodel/person_detect_yolov5n.kmodel"
person_labels = ["person"]
person_anchors = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]

hand_det_kmodel_path = "/sdcard/examples/kmodel/hand_det.kmodel"
hand_kp_kmodel_path = "/sdcard/examples/kmodel/handkp_det.kmodel"
hand_labels = ["hand"]
hand_anchors = [26, 27, 53, 52, 75, 71, 80, 99, 106, 82, 99, 134, 140, 113, 161, 172, 245, 276]

# =========================
# 初始化 PipeLine
# =========================
pl = PipeLine(rgb888p_size=rgb888p_size, display_size=display_size, display_mode=display_mode)
pl.create(Sensor(width=rgb888p_size[0], height=rgb888p_size[1]))

# =========================
# 初始化检测器
# =========================
person_det = PersonDetectionApp(
    person_kmodel_path,
    model_input_size=[640, 640],
    labels=person_labels,
    anchors=person_anchors,
    confidence_threshold=0.2,
    nms_threshold=0.6,
    nms_option=False,
    strides=[8, 16, 32],
    rgb888p_size=rgb888p_size,
    display_size=display_size,
    debug_mode=0
)
person_det.config_preprocess()

hand_det = HandDetApp(
    hand_det_kmodel_path,
    labels=hand_labels,
    model_input_size=[512, 512],
    anchors=hand_anchors,
    confidence_threshold=0.2,
    nms_threshold=0.5,
    nms_option=False,
    strides=[8, 16, 32],
    rgb888p_size=rgb888p_size,
    display_size=display_size,
    debug_mode=0
)
hand_det.config_preprocess()

hand_kp = HandKPClassApp(
    hand_kp_kmodel_path,
    model_input_size=[256, 256],
    rgb888p_size=rgb888p_size,
    display_size=display_size
)

locker = TargetLocker(display_size, rgb888p_size)

x_pid.set_target(display_size[0] / 2)
y_pid.set_target(display_size[1] / 2)

# =========================
# 状态变量
# =========================
tracking_enabled = False
gesture_count = 0
last_gesture = None

sm_cx = None
sm_cy = None
y_out_smooth = 0.0
last_servo_ms = utime.ticks_ms()

clock = time.clock()
frame = 0

# =========================
# 主循环
# =========================
while True:
    clock.tick()
    frame += 1

    img = pl.get_frame()
    pl.osd_img.clear()

    # === 人体检测 ===
    person_dets = person_det.run(img)

    # === 手势检测（每帧都检测）===
    hand_dets = hand_det.run(img)
    current_gesture = None

    for hdet in hand_dets:
        x1, y1, x2, y2 = hdet[2], hdet[3], hdet[4], hdet[5]
        w, h = int(x2 - x1), int(y2 - y1)

        if h < (0.05 * rgb888p_size[1]):
            continue

        hand_kp.config_preprocess(hdet)
        gesture = hand_kp.run(img)

        if gesture:
            current_gesture = gesture
            # 显示手势
            hx = int(x1 * display_size[0] // rgb888p_size[0])
            hy = int(y1 * display_size[1] // rgb888p_size[1])
            hw = int(w * display_size[0] // rgb888p_size[0])
            hh = int(h * display_size[1] // rgb888p_size[1])
            pl.osd_img.draw_rectangle(hx, hy, hw, hh, color=(255, 0, 255, 255), thickness=2)
            pl.osd_img.draw_string_advanced(hx, hy - 30, 24, gesture, color=(255, 0, 255, 255))
        break

    # === 手势控制逻辑 ===
    if current_gesture is not None:
        if current_gesture == last_gesture:
            gesture_count += 1
        else:
            gesture_count = 1
            last_gesture = current_gesture
    else:
        # 没有检测到手势时不重置，保持一段时间
        pass

    # 触发手势控制
    if gesture_count >= GESTURE_HOLD_FRAMES:
        if last_gesture == GESTURE_START and not tracking_enabled:
            tracking_enabled = True
            locker.reset()
            sm_cx = None
            sm_cy = None
            print(">>> TRACKING ON!")
            gesture_count = 0
        elif last_gesture == GESTURE_STOP and tracking_enabled:
            tracking_enabled = False
            locker.reset()
            print(">>> TRACKING OFF!")
            gesture_count = 0

    # === 跟踪逻辑 ===
    locked_index = -1

    if tracking_enabled:
        locked_index, center = locker.update(person_dets)

        if locked_index >= 0 and center is not None:
            cx, cy = center[0], center[1]

            sm_cx = ema(sm_cx, cx, EMA_ALPHA_X)
            sm_cy = ema(sm_cy, cy, EMA_ALPHA_Y)

            now_ms = utime.ticks_ms()
            if utime.ticks_diff(now_ms, last_servo_ms) >= SERVO_UPDATE_MS:
                last_servo_ms = now_ms

                x_output = x_pid.update(sm_cx)
                y_output = y_pid.update(sm_cy)

                y_out_smooth = Y_OUT_EMA_ALPHA * y_output + (1.0 - Y_OUT_EMA_ALPHA) * y_out_smooth
                y_out_smooth *= y_soft_limit_scale(y_angle)

                x_output = clamp(x_output, -MAX_STEP_X_DEG, MAX_STEP_X_DEG)
                y_output_final = clamp(y_out_smooth, -MAX_STEP_Y_DEG, MAX_STEP_Y_DEG)

                if abs(x_output) < MIN_STEP_X_DEG:
                    x_output = 0
                if abs(y_output_final) < MIN_STEP_Y_DEG:
                    y_output_final = 0

                if x_output != 0:
                    x_angle = clamp(x_angle + x_output, X_MIN, X_MAX)
                    servo_x.position(0, round(x_angle, 1))

                if y_output_final != 0:
                    y_angle = clamp(y_angle - y_output_final, Y_MIN, Y_MAX)
                    servo_y.position(1, round(y_angle, 1))
        else:
            if locker.lost == 1:
                x_pid.reset()
                y_pid.reset()
                y_out_smooth = 0.0
                sm_cx = None
                sm_cy = None

    # === 绘制人体框 ===
    person_det.draw_result(pl, person_dets, locked_index)

    # === 状态显示 ===
    if tracking_enabled:
        pl.osd_img.draw_string_advanced(10, 10, 28, "TRACKING ON", color=(0, 255, 0, 255))
    else:
        pl.osd_img.draw_string_advanced(10, 10, 28, "thumbUp to START", color=(255, 255, 0, 255))

    # 显示手势计数（调试）
    if last_gesture:
        pl.osd_img.draw_string_advanced(10, 450, 20, "Gesture: " + str(last_gesture) + " (" + str(gesture_count) + ")", color=(255, 255, 255, 255))

    pl.show_image()

    if frame % 15 == 0:
        gc.collect()

    if frame % 30 == 0:
        print("FPS:", clock.fps(), "persons:", len(person_dets) if person_dets else 0, "lock:", locked_index, "gesture:", last_gesture)
