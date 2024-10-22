<p align="center">
  <img src="assets/banner-YOLO.png" align="middle" width = "1000" />
</p>

English | [简体中文](README_cn.md)


## YOLOv6-Segmentation

Implementation of Instance Segmentation based on [YOLOv6 v4.0 code](https://github.com/meituan/YOLOv6/tree/main).

<p align="left">
  <img src="assets/speed_comparision_seg.png" align="center", width="500"/>
  <br>
</p>


## What's New
- [2023.09.15] Release [YOLOv6-Segmentation](https://github.com/meituan/YOLOv6/tree/yolov6-seg). 🚀 [Performance](https://github.com/meituan/YOLOv6/tree/yolov6-seg#yolov6-segmentation)
- [2023.04.28] Release [YOLOv6Lite](configs/yolov6_lite/README.md) models on mobile or CPU. ⭐️ [Mobile Benchmark](#Mobile-Benchmark)
- [2023.03.10] Release [YOLOv6-Face](https://github.com/meituan/YOLOv6/tree/yolov6-face). 🔥 [Performance](https://github.com/meituan/YOLOv6/tree/yolov6-face#performance-on-widerface)
- [2023.03.02] Update [base models](configs/base/README.md) to version 3.0.
- [2023.01.06] Release P6 models and enhance the performance of P5 models. ⭐️ [Benchmark](#Benchmark)
- [2022.11.04] Release [base models](configs/base/README.md) to simplify the training and deployment process.
- [2022.09.06] Customized quantization methods. 🚀 [Quantization Tutorial](./tools/qat/README.md)
- [2022.09.05] Release M/L models and update N/T/S models with enhanced performance.
- [2022.06.23] Release N/T/S models with excellent performance.

## Object Detection Benchmark
| Model                                                        | Size | mAP<sup>val<br/>0.5:0.95 | Speed<sup>T4<br/>trt fp16 b1 <br/>(fps) | Speed<sup>T4<br/>trt fp16 b32 <br/>(fps) | Params<br/><sup> (M) | FLOPs<br/><sup> (G) |
| :----------------------------------------------------------- | ---- | :----------------------- | --------------------------------------- | ---------------------------------------- | -------------------- | ------------------- |
| [**YOLOv6-N**](https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6n.pt) | 640  | 37.5                     | 779                                     | 1187                                     | 4.7                  | 11.4                |
| [**YOLOv6-S**](https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6s.pt) | 640  | 45.0                     | 339                                     | 484                                      | 18.5                 | 45.3                |
| [**YOLOv6-M**](https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6m.pt) | 640  | 50.0                     | 175                                     | 226                                      | 34.9                 | 85.8                |
| [**YOLOv6-L**](https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6l.pt) | 640  | 52.8                     | 98                                      | 116                                      | 59.6                 | 150.7               |
|                              |                               |                                |                    |                        |
| [**YOLOv6-N6**](https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6n6.pt) | 1280 | 44.9                     | 228                                     | 281                                      | 10.4                 | 49.8                |
| [**YOLOv6-S6**](https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6s6.pt) | 1280 | 50.3                     | 98                                      | 108                                      | 41.4                 | 198.0               |
| [**YOLOv6-M6**](https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6m6.pt) | 1280 | 55.2                     | 47                                      | 55                                       | 79.6                 | 379.5               |
| [**YOLOv6-L6**](https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6l6.pt) | 1280 | 57.2                     | 26                                      | 29                                       | 140.4                | 673.4               |
<details>
<summary>Table Notes</summary>

- All checkpoints are trained with self-distillation except for YOLOv6-N6/S6 models trained to 300 epochs without distillation.
- Results of the mAP and speed are evaluated on [COCO val2017](https://cocodataset.org/#download) dataset with the input resolution of 640×640 for P5 models and 1280x1280 for P6 models.
- Speed is tested with TensorRT 7.2 on T4.
- Refer to [Test speed](./docs/Test_speed.md) tutorial to reproduce the speed results of YOLOv6.
- Params and FLOPs of YOLOv6 are estimated on deployed models.

## Segmentation Benchmark
| Model                                                        | Size | mAP<sup>box<br/>50-95 | mAP<sup>mask<br/>50-95 | Speed<sup>T4<br/>trt fp16 b1 <br/>(fps) | Params<br/><sup> (M) | FLOPs<br/><sup> (G) |
| :----------------------------------------------------------- | ---- | :-------------------- | ---------------------- | --------------------------------------- | -------------------- | ------------------- |
| [**YOLOv6-N**](https://github.com/meituan/YOLOv6/releases/download/0.4.1/yolov6n_seg.pt) | 640  | 35.3                  | 31.2                   | 645                                     | 4.9                  | 7.0                 |
| [**YOLOv6-S**](https://github.com/meituan/YOLOv6/releases/download/0.4.1/yolov6s_seg.pt) | 640  | 44.0                  | 38.0                   | 292                                     | 19.6                 | 27.7                |
| [**YOLOv6-M**](https://github.com/meituan/YOLOv6/releases/download/0.4.1/yolov6m_seg.pt) | 640  | 48.2                  | 41.3                   | 148                                     | 37.1                 | 54.3                |
| [**YOLOv6-L**](https://github.com/meituan/YOLOv6/releases/download/0.4.1/yolov6l_seg.pt) | 640  | 51.1                  | 43.7                   | 93                                      | 63.6                 | 95.5                |
| [**YOLOv6-X**](https://github.com/meituan/YOLOv6/releases/download/0.4.1/yolov6x_seg.pt) | 640  | 52.2                  | 44.8                   | 47                                      | 119.1                | 175.5               |

#### Table Notes
- All checkpoints are trained from scratch on COCO for 300 epochs without distillation.
- Results of the mAP and speed are evaluated on [COCO val2017](https://cocodataset.org/#download) dataset with the input resolution of 640×640.
- Speed is tested with TensorRT 8.5 on T4 without post-processing.


## Quick Start
<details open>
<summary> Install</summary>


```shell
git clone https://github.com/meituan/YOLOv6
cd YOLOv6
git checkout yolov6-seg
pip install -r requirements.txt
```
</details>

<details open>
<summary> Training </summary>

Single GPU

```shell
python tools/train.py --batch 8 --conf configs/yolov6s_seg_finetune.py --data data/coco.yaml --device 0
```

Multi GPUs (DDP mode recommended)

```shell
python -m torch.distributed.launch --nproc_per_node 8 tools/train.py --batch 64 --conf configs/yolov6s_seg_finetune.py --data data/coco.yaml --device 0,1,2,3,4,5,6,7
```
- fuse_ab: Not supported in current version
- conf: select config file to specify network/optimizer/hyperparameters. We recommend to apply yolov6n/s/m/l_finetune.py when training on your custom dataset.
- data: prepare dataset and specify dataset paths in data.yaml ( [COCO](http://cocodataset.org), [YOLO format coco labels](https://github.com/meituan/YOLOv6/releases/download/0.4.1/coco2017labels-segments.zip) )
- make sure your dataset structure as follows:
```
├── coco
│   ├── annotations
│   │   ├── instances_train2017.json
│   │   └── instances_val2017.json
│   ├── images
│   │   ├── train2017
│   │   └── val2017
│   ├── labels
│   │   ├── train2017
│   │   ├── val2017
│   ├── LICENSE
│   ├── README.txt
```


</details>


<details open>
<summary> Evaluation</summary>

Reproduce mAP on COCO val2017 dataset with 640×640 resolution

```shell
python tools/eval.py --data data/coco.yaml --batch 32 --weights yolov6s_seg.pt --task val
```

</details>


<details>
<summary>Inference</summary>

First, download a pretrained model from the YOLOv6 [release](https://github.com/meituan/YOLOv6/releases/tag/0.4.1) or use your trained model to do inference.

Second, run inference with `tools/infer.py`

```shell
python tools/infer.py --weights yolov6s_seg.pt --source img.jpg / imgdir / video.mp4
```
If you want to inference on local camera or  web camera, you can run:
```shell
python tools/infer.py --weights yolov6s_seg.pt --webcam --webcam-addr 0
```
`webcam-addr` can be local camera number id or rtsp address.
Maybe you want to eval a solo-head model, remember to add the *--issolo* parameter.

</details>

<details>
<summary> Deployment</summary>

*  [ONNX](./deploy/ONNX)
*  [OpenCV Python/C++](./deploy/ONNX/OpenCV)
*  [OpenVINO](./deploy/OpenVINO)
*  [TensorRT](./deploy/TensorRT)
*  [NCNN](./deploy/NCNN)
*  [Android](./deploy/NCNN/Android)
</details>

<details open>
<summary> Tutorials</summary>

*  [User Guide(zh_CN)](https://yolov6-docs.readthedocs.io/zh_CN/latest/)
*  [Train COCO Dataset](./docs/Train_coco_data.md)
*  [Train custom data](./docs/Train_custom_data.md)
*  [Test speed](./docs/Test_speed.md)
*  [Tutorial of Quantization for YOLOv6](./docs/Tutorial%20of%20Quantization.md)
</details>

<details>
<summary> Third-party resources</summary>

 * YOLOv6 Training with Amazon Sagemaker: [yolov6-sagemaker](https://github.com/ashwincc/yolov6-sagemaker) from [ashwincc](https://github.com/ashwincc)  

 * YOLOv6 NCNN Android app demo: [ncnn-android-yolov6](https://github.com/FeiGeChuanShu/ncnn-android-yolov6) from [FeiGeChuanShu](https://github.com/FeiGeChuanShu)

 * YOLOv6 ONNXRuntime/MNN/TNN C++: [YOLOv6-ORT](https://github.com/DefTruth/lite.ai.toolkit/blob/main/lite/ort/cv/yolov6.cpp), [YOLOv6-MNN](https://github.com/DefTruth/lite.ai.toolkit/blob/main/lite/mnn/cv/mnn_yolov6.cpp) and [YOLOv6-TNN](https://github.com/DefTruth/lite.ai.toolkit/blob/main/lite/tnn/cv/tnn_yolov6.cpp) from [DefTruth](https://github.com/DefTruth)

 * YOLOv6 TensorRT Python: [yolov6-tensorrt-python](https://github.com/Linaom1214/TensorRT-For-YOLO-Series) from [Linaom1214](https://github.com/Linaom1214)

 * YOLOv6 TensorRT Windows C++: [yolort](https://github.com/zhiqwang/yolov5-rt-stack/tree/main/deployment/tensorrt-yolov6) from [Wei Zeng](https://github.com/Wulingtian)

 * [YOLOv6 web demo](https://huggingface.co/spaces/nateraw/yolov6) on [Huggingface Spaces](https://huggingface.co/spaces) with [Gradio](https://github.com/gradio-app/gradio). [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/nateraw/yolov6)

 * [Interactive demo](https://yolov6.dagshubusercontent.com/) on [DagsHub](https://dagshub.com) with [Streamlit](https://github.com/streamlit/streamlit)

 * Tutorial: [How to train YOLOv6 on a custom dataset](https://blog.roboflow.com/how-to-train-yolov6-on-a-custom-dataset/) <a href="https://colab.research.google.com/drive/1YnbqOinBZV-c9I7fk_UL6acgnnmkXDMM"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

 * YouTube Tutorial: [How to train YOLOv6 on a custom dataset](https://youtu.be/fFCWrMFH2UY)

 * Demo of YOLOv6 inference on Google Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mahdilamb/YOLOv6/blob/main/inference.ipynb)

 * Blog post: [YOLOv6 Object Detection – Paper Explanation and Inference](https://learnopencv.com/yolov6-object-detection/)

   </details>

### [FAQ（Continuously updated）](https://github.com/meituan/YOLOv6/wiki/FAQ%EF%BC%88Continuously-updated%EF%BC%89)

If you have any questions, welcome to join our WeChat group to discuss and exchange.
<p align="center">
  <img src="assets/wechat_qrcode.png" align="middle" width = "1000" />
</p>