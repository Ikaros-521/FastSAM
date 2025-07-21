from ultralytics import YOLO
import gradio as gr
import torch
from utils.tools_gradio import fast_process
from utils.tools import format_results, box_prompt, point_prompt, text_prompt
from PIL import ImageDraw
import numpy as np
import os

import torch.serialization
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.nn.modules import Conv
torch.serialization.add_safe_globals([torch.nn.modules.container.Sequential])
torch.serialization.safe_globals([Conv])
torch.serialization.add_safe_globals([SegmentationModel])

# Load the pre-trained model
model = YOLO('./weights/FastSAM-s.pt')

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Description
title = "<center><strong><font size='8'>🏃 快速分割任意物体 🤗</font></strong></center>"

news = """ # 📖 新闻
        🔥 2023/07/14: 在文本模式下添加了“更宽结果”按钮（感谢 [gaoxinge](https://github.com/CASIA-IVA-Lab/FastSAM/pull/95)）。

        🔥 2023/06/29: 支持文本模式（感谢 [gaoxinge](https://github.com/CASIA-IVA-Lab/FastSAM/pull/47)）。

        🔥 2023/06/26: 支持点模式。（更好更快的交互即将到来！）

        🔥 2023/06/24: 在全分割模式下添加“高级选项”，以获得更详细的调整。        
        """  

description_e = """这是 Github 项目 🏃 [Fast Segment Anything Model](https://github.com/CASIA-IVA-Lab/FastSAM) 的演示。欢迎为其点亮一颗星 ⭐️。
                
                🎯 上传一张图片，使用 Fast Segment Anything 进行分割（全分割模式）。其他模式即将上线。
                
                ⌛️ 生成分割结果大约需要 6 秒。队列的并发数为 1，人多时请耐心等待。
                
                🚀 想要更快的结果，可以使用更小的输入尺寸，并取消勾选高视觉质量。
                
                📣 你也可以通过此 Colab 获得任意图片的分割结果：[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oX14f6IneGGw612WgVlAiy91UHwFAvr9?usp=sharing)
                
                😚 非常感谢 @HuggingFace 团队为我们提供 GPU 支持。
                
                🏠 查看我们的 [模型卡 🏃](https://huggingface.co/An-619/FastSAM)
                
              """

description_p = """ # 🎯 点模式使用说明
                这是 Github 项目 🏃 [Fast Segment Anything Model](https://github.com/CASIA-IVA-Lab/FastSAM) 的演示。欢迎为其点亮一颗星 ⭐️。
                
                1. 上传图片或选择示例。
                
                2. 选择点标签（“添加掩码”表示正点，“移除区域”表示不分割的负点）。
                
                3. 在图片上逐个添加点。
                
                4. 点击“点提示分割”按钮获取分割结果。
                
                **5. 如果出错，点击“清除点”按钮后重试可能有帮助。**
                
              """

examples = [["examples/sa_8776.jpg"], ["examples/sa_414.jpg"], ["examples/sa_1309.jpg"], ["examples/sa_11025.jpg"],
            ["examples/sa_561.jpg"], ["examples/sa_192.jpg"], ["examples/sa_10039.jpg"], ["examples/sa_862.jpg"]]

default_example = examples[0]

css = "h1 { text-align: center } .about { text-align: justify; padding-left: 10%; padding-right: 10%; }"


def segment_everything(
    input,
    input_size=1024, 
    iou_threshold=0.7,
    conf_threshold=0.25,
    better_quality=False,
    withContours=True,
    use_retina=True,
    text="",
    wider=False,
    mask_random_color=True,
):
    print("Segmenting everything...")

    input_size = int(input_size)  # 确保 imgsz 是整数
    # Thanks for the suggestion by hysts in HuggingFace.
    w, h = input.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    input = input.resize((new_w, new_h))

    print("即将加载模型，请稍等...")

    results = model(input,
                    device=device,
                    retina_masks=True,
                    iou=iou_threshold,
                    conf=conf_threshold,
                    imgsz=input_size,)

    print("模型加载完成，开始处理图像...")

    if len(text) > 0:
        results = format_results(results[0], 0)
        annotations, _ = text_prompt(results, text, input, device=device, wider=wider)
        annotations = np.array([annotations])
    else:
        annotations = results[0].masks.data
    
    print("图像处理完成，开始绘制结果...")

    fig = fast_process(annotations=annotations,
                       image=input,
                       device=device,
                       scale=(1024 // input_size),
                       better_quality=better_quality,
                       mask_random_color=mask_random_color,
                       bbox=None,
                       use_retina=use_retina,
                       withContours=withContours,)
    
    print(f"type(fig): {type(fig)}")
    
    return fig


def segment_with_points(
    input,
    input_size=1024, 
    iou_threshold=0.7,
    conf_threshold=0.25,
    better_quality=False,
    withContours=True,
    use_retina=True,
    mask_random_color=True,
):
    global global_points
    global global_point_label
    
    input_size = int(input_size)  # 确保 imgsz 是整数
    # Thanks for the suggestion by hysts in HuggingFace.
    w, h = input.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    input = input.resize((new_w, new_h))
    
    scaled_points = [[int(x * scale) for x in point] for point in global_points]

    results = model(input,
                    device=device,
                    retina_masks=True,
                    iou=iou_threshold,
                    conf=conf_threshold,
                    imgsz=input_size,)
    
    results = format_results(results[0], 0)
    annotations, _ = point_prompt(results, scaled_points, global_point_label, new_h, new_w)
    annotations = np.array([annotations])

    fig = fast_process(annotations=annotations,
                       image=input,
                       device=device,
                       scale=(1024 // input_size),
                       better_quality=better_quality,
                       mask_random_color=mask_random_color,
                       bbox=None,
                       use_retina=use_retina,
                       withContours=withContours,)

    # print(f"type: {type(fig)}")

    global_points = []
    global_point_label = []
    return fig


def get_points_with_draw(image, label, evt: gr.SelectData):
    global global_points
    global global_point_label

    x, y = evt.index[0], evt.index[1]
    point_radius, point_color = 15, (255, 255, 0) if label == '添加掩码' else (255, 0, 255)
    global_points.append([x, y])
    global_point_label.append(1 if label == '添加掩码' else 0)
    
    print(x, y, label == '添加掩码')
    
    # 创建一个可以在图像上绘图的对象
    draw = ImageDraw.Draw(image)
    draw.ellipse([(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)], fill=point_color)
    return image


cond_img_e = gr.Image(label="输入图片", value=default_example[0], type='pil')
cond_img_p = gr.Image(label="带点的输入图片", value=default_example[0], type='pil')
cond_img_t = gr.Image(label="带文本的输入图片", value="examples/dogs.jpg", type='pil')

segm_img_e = gr.Image(label="分割结果", interactive=False, type='pil')
segm_img_p = gr.Image(label="点分割结果", interactive=False, type='pil')
segm_img_t = gr.Image(label="文本分割结果", interactive=False, type='pil')

global_points = []
global_point_label = []

input_size_slider = gr.components.Slider(minimum=512,
                                         maximum=1024,
                                         value=1024,
                                         step=64,
                                         label='输入尺寸',
                                         info='我们的模型训练尺寸为 1024')

with gr.Blocks(css=css, title='Fast Segment Anything') as demo:
    with gr.Row():
        with gr.Column(scale=1):
            # Title
            gr.Markdown(title)

        with gr.Column(scale=1):
            # News
            gr.Markdown(news)

    with gr.Tab("全分割模式"):
        # Images
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                cond_img_e.render()

            with gr.Column(scale=1):
                segm_img_e.render()

        # Submit & Clear
        with gr.Row():
            with gr.Column():
                input_size_slider.render()

                with gr.Row():
                    contour_check = gr.Checkbox(value=True, label='显示边缘', info='绘制掩码边缘')

                    with gr.Column():
                        segment_btn_e = gr.Button("一键全分割", variant='primary')
                        clear_btn_e = gr.Button("清除", variant="secondary")

            with gr.Column():
                with gr.Accordion("高级选项", open=False):
                    iou_threshold = gr.Slider(0.1, 0.9, 0.7, step=0.1, label='iou 阈值', info='用于过滤注释的 iou 阈值')
                    conf_threshold = gr.Slider(0.1, 0.9, 0.25, step=0.05, label='置信度阈值', info='目标置信度阈值')
                    with gr.Row():
                        mor_check = gr.Checkbox(value=False, label='更好视觉质量', info='使用 morphologyEx 获得更好质量')
                        with gr.Column():
                            retina_check = gr.Checkbox(value=True, label='高分辨率掩码', info='绘制高分辨率分割掩码')

                # Description
                gr.Markdown(description_e)

    segment_btn_e.click(segment_everything,
                        inputs=[
                            cond_img_e,
                            input_size_slider,
                            iou_threshold,
                            conf_threshold,
                            mor_check,
                            contour_check,
                            retina_check,
                        ],
                        outputs=segm_img_e)

    with gr.Tab("点模式"):
        # Images
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                cond_img_p.render()

            with gr.Column(scale=1):
                segm_img_p.render()
                
        # Submit & Clear
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    add_or_remove = gr.Radio(["添加掩码", "移除区域"], value="添加掩码", label="点标签（前景/背景）")

                    with gr.Column():
                        segment_btn_p = gr.Button("点提示分割", variant='primary')
                        clear_btn_p = gr.Button("清除点", variant='secondary')

            with gr.Column():
                # Description
                gr.Markdown(description_p)

    cond_img_p.select(get_points_with_draw, [cond_img_p, add_or_remove], cond_img_p)

    segment_btn_p.click(segment_with_points,
                        inputs=[cond_img_p],
                        outputs=[segm_img_p])

    with gr.Tab("文本模式"):
        # Images
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                cond_img_t.render()

            with gr.Column(scale=1):
                segm_img_t.render()

        # Submit & Clear
        with gr.Row():
            with gr.Column():
                input_size_slider_t = gr.components.Slider(minimum=512,
                                                           maximum=1024,
                                                           value=1024,
                                                           step=64,
                                                           label='输入尺寸',
                                                           info='我们的模型训练尺寸为 1024')
                with gr.Row():
                    with gr.Column():
                        contour_check = gr.Checkbox(value=True, label='显示边缘', info='绘制掩码边缘')
                        text_box = gr.Textbox(label="文本提示", value="a black dog")

                    with gr.Column():
                        segment_btn_t = gr.Button("文本分割", variant='primary')
                        clear_btn_t = gr.Button("清除", variant="secondary")

            with gr.Column():
                with gr.Accordion("高级选项", open=False):
                    iou_threshold = gr.Slider(0.1, 0.9, 0.7, step=0.1, label='iou 阈值', info='用于过滤注释的 iou 阈值')
                    conf_threshold = gr.Slider(0.1, 0.9, 0.25, step=0.05, label='置信度阈值', info='目标置信度阈值')
                    with gr.Row():
                        mor_check = gr.Checkbox(value=False, label='更好视觉质量', info='使用 morphologyEx 获得更好质量')
                        retina_check = gr.Checkbox(value=True, label='高分辨率掩码', info='绘制高分辨率分割掩码')
                        wider_check = gr.Checkbox(value=False, label='更宽结果', info='更宽的分割结果')

                # Description
                gr.Markdown(description_e)
    
    segment_btn_t.click(segment_everything,
                        inputs=[
                            cond_img_t,
                            input_size_slider_t,
                            iou_threshold,
                            conf_threshold,
                            mor_check,
                            contour_check,
                            retina_check,
                            text_box,
                            wider_check,
                        ],
                        outputs=segm_img_t)

    def clear():
        return None, None
    
    def clear_text():
        return None, None, None

    clear_btn_e.click(clear, outputs=[cond_img_e, segm_img_e])
    clear_btn_p.click(clear, outputs=[cond_img_p, segm_img_p])
    clear_btn_t.click(clear_text, outputs=[cond_img_p, segm_img_p, text_box])

demo.queue()
demo.launch(server_name="0.0.0.0", share=False)
