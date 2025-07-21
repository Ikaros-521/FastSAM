import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO  # 或 from fastsam.model import FastSAM
from fastsam.prompt import FastSAMPrompt
from utils.tools import format_results, point_prompt
import argparse
import time

# 配置
VIDEO_PATH = '720p.mp4'  # 输入视频路径
MODEL_PATH = './weights/FastSAM-s.pt'  # 权重路径
INPUT_SIZE = 1024  # 输入分辨率
IOU_THRESHOLD = 0.7
CONF_THRESHOLD = 0.25
BETTER_QUALITY = False
WITH_CONTOURS = True
USE_RETINA = True

# argparse参数
parser = argparse.ArgumentParser(description='视频点分割')
parser.add_argument('--show-fps', action='store_true', help='是否显示FPS')
parser.add_argument('--bg-color', type=str, default='black', choices=['black', 'green'], help='背景色: black或green')
args = parser.parse_args()
SHOW_FPS = args.show_fps
BG_COLOR = args.bg_color

# 全局变量
points = []
labels = []  # 1:前景, 0:背景
drawing = True
first_frame = None
clone = None

def mouse_callback(event, x, y, flags, param):
    global points, labels, first_frame, clone, model, device, scale

    if event == cv2.EVENT_LBUTTONDOWN:
        print("左键点击")
        points.append([x, y])
        labels.append(1)
    elif event == cv2.EVENT_RBUTTONDOWN:
        print("右键点击")
        points.append([x, y])
        labels.append(0)
    else:
        # 不是点击事件，直接返回，不做后续处理
        return

    # 只有点击时才会走到这里
    if points:
        # 计算缩放后的点
        scaled_points = [[int(px * scale), int(py * scale)] for px, py in points]
        mask = get_mask(model, first_frame, scaled_points, labels, device)
        mask = 255 - mask  # 反转掩码，确保选中的区域变亮
        vis = cv2.addWeighted(first_frame, 0.6, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.4, 0)
        # 画点
        for (px, py), label in zip(points, labels):
            color = (0, 255, 255) if label == 1 else (255, 0, 255)
            cv2.circle(vis, (px, py), 8, color, -1)
        cv2.imshow('First Frame', vis)
    else:
        cv2.imshow('First Frame', first_frame)

def preprocess_frame(frame, input_size=INPUT_SIZE):
    h, w = frame.shape[:2]
    scale = input_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    return resized, scale

def get_mask(model, frame, points, labels, device, input_size=INPUT_SIZE, iou_threshold=IOU_THRESHOLD, conf_threshold=CONF_THRESHOLD, use_retina=USE_RETINA):
    
    # resize与gradio一致
    h0, w0 = frame.shape[:2]
    scale = input_size / max(w0, h0)
    new_w, new_h = int(w0 * scale), int(h0 * scale)
    frame_resized = cv2.resize(frame, (new_w, new_h))
    # 点也要缩放
    scaled_points = [[int(x * scale), int(y * scale)] for x, y in points]
    input_pil = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
    
    # 记录起始时间
    start_time = time.time()
    
    results = model(
        input_pil,
        device=device,
        retina_masks=use_retina,
        iou=iou_threshold,
        conf=conf_threshold,
        imgsz=input_size
    )

    # 记录结束时间
    end_time = time.time()

    # 计算耗时
    elapsed_time = end_time - start_time
    print(f"模型处理耗时: {elapsed_time:.3f}秒")

    results = format_results(results[0], 0)
    mask, _ = point_prompt(results, scaled_points, labels, new_h, new_w)
    mask = mask.astype(np.uint8) * 255
    # 返回resize到原图尺寸的掩码
    mask = cv2.resize(mask, (w0, h0), interpolation=cv2.INTER_NEAREST)

    return mask

# main函数内相关变量声明提前到全局
model = None
scale = 1.0
device = None

def main():
    global points, labels, first_frame, clone, model, device, scale

    # 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    # 加载模型
    model = YOLO(MODEL_PATH)

    # 打开视频
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    if not ret:
        print("无法读取视频")
        return

    # 预处理第一帧（不resize，保持原图，resize在get_mask里做）
    first_frame = frame.copy()
    clone = frame.copy()

    cv2.namedWindow('First Frame')
    cv2.setMouseCallback('First Frame', mouse_callback)

    print("请在第一帧上打点，左键前景，右键背景，按c清空，回车开始分割")

    # 打点阶段
    while True:
        # 如果没有点，主动刷新显示原始图像
        if not points:
            cv2.imshow('First Frame', first_frame)
        key = cv2.waitKey(1)
        if key == ord('c'):
            first_frame = clone.copy()
            points = []
            labels = []
            cv2.imshow('First Frame', first_frame)
        elif key == 13:  # 回车
            if len(points) == 0:
                print("请先打点！")
                continue
            break
        elif key == 27:  # ESC
            cap.release()
            cv2.destroyAllWindows()
            return

    # 生成第一帧掩码
    mask = get_mask(model, first_frame, points, labels, device)

    # 分割视频流
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 回到第一帧
    print("开始分割视频流，ESC退出")
    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 用同样的点分割每一帧
        mask = get_mask(model, frame, points, labels, device)
        # 只保留目标区域
        if BG_COLOR == 'green':
            bg = np.full_like(frame, (0, 255, 0))  # 绿色背景
        else:
            bg = np.zeros_like(frame)  # 黑色背景
        result = np.where(mask[..., None].astype(bool), frame, bg)
        # FPS显示
        if SHOW_FPS:
            now = time.time()
            fps = 1.0 / (now - prev_time)
            prev_time = now
            cv2.putText(result, f"FPS: {fps:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Segmented Video', result)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 