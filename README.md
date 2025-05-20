UA-DETRAC Vehicle Detection: Comparative Performance Report Using YOLOv8 and Faster R-CNN
Author: Ömer Dursun Cengizhan Kınay Barış Başaran

________________________________________
1. Introduction
This report presents a comprehensive and detailed comparative analysis of object detection techniques applied to the UA-DETRAC dataset, focusing on two state-of-the-art methods:
1.	YOLOv8 (Nano and Small variants)
2.	Faster R-CNN with ResNet-50 Feature Pyramid Network (FPN)
The objectives of this study were to:
•	Preprocess and structure the dataset for modern deep learning models
•	Train and evaluate multiple object detection architectures
•	Compare metrics including mAP, precision, recall, and inference speed
•	Iterate on training parameters and architecture variants to achieve optimal performance
________________________________________
2. Dataset Description
2.1 Dataset Overview
The UA-DETRAC dataset is a widely-used traffic surveillance benchmark composed of high-resolution video sequences. It includes detailed annotations for detecting and tracking multiple vehicle categories under various environmental conditions. For this project, video frames were extracted into still images and used with corresponding bounding box annotations.
2.2 Class Definitions
The four categories annotated in this study are:
•	Car
•	Van
•	Bus
•	Others
2.3 Label Format
Each image has an associated .txt annotation file in YOLO format:
<class_id> <x_center> <y_center> <width> <height>
All coordinates are normalized between 0 and 1.
2.4 Dataset Structure
The following structure was used:
YoloV8_detract/
├── datasets/
│   ├── ua_detrac/
│       ├── images/
│       │   ├── train/
│       │   └── val/
│       ├── labels/
│       │   ├── train/
│       │   └── val/
│       └── data.yaml
Each data.yaml file specified class names, training and validation image paths.
________________________________________
3. Environment and Setup
3.1 Virtual Environment
•	OS: Windows 11
•	Python version: 3.12.10
•	Virtual Environment: python -m venv venv
•	Activated via: .\venv\Scripts\activate
3.2 Dependencies
pip install ultralytics torch torchvision matplotlib
3.3 Hardware
•	GPU: NVIDIA GeForce RTX 3080 Ti
________________________________________
4. YOLOv8 Implementation
4.1 YOLOv8n (Baseline)
yolo task=detect mode=train model=yolov8n.pt data=datasets/ua_detrac/data.yaml epochs=50 imgsz=640 device=0
•	mAP@0.5: 0.635
•	mAP@0.5:0.95: 0.487
•	Precision: 0.622
•	Recall: 0.631
4.2 YOLOv8s (With Augmentation)
yolo task=detect mode=train model=yolov8s.pt data=datasets/ua_detrac/data.yaml imgsz=960 epochs=30 patience=10 device=0 mosaic=1.0 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4
•	Early Stopping triggered at Epoch 19
•	mAP@0.5: 0.697
•	mAP@0.5:0.95: 0.541
•	Precision: 0.678
•	Recall: 0.666
  
 
4.3 Insights
•	Increasing image size (640 → 960) improved detection accuracy
•	Mosaic and HSV-based augmentation provided robustness to variance
•	YOLOv8s significantly outperformed YOLOv8n, especially on small and medium objects
•	Early stopping was used to terminate training at the optimal epoch and avoid overfitting
________________________________________
5. Faster R-CNN Implementation
5.1 Architecture
Used fasterrcnn_resnet50_fpn from torchvision.models.detection. Modified the classification head:
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=4)
5.2 Dataset Loading
Custom dataset loader was implemented to convert YOLO labels to absolute coordinates:
xmin = (x_center - width/2) * image_width
All bounding boxes and class labels were stored in dictionaries compatible with torchvision models.
5.3 Training Configuration
Setting	Value
Epochs	50
Batch Size	4
Optimizer	SGD
Learning Rate	0.005
Device	CUDA (RTX 3080 Ti)
	
5.4 Evaluation Results
•	mAP@0.5: 0.483
•	mAP@0.5:0.95: 0.351
•	Bus AP: 0.546

 



________________________________________



6. Model Progression and Training Summary
Model	mAP@0.5	mAP@0.5:0.95	Precision	Recall	Notes
Faster R-CNN	0.483	0.351	~0.62	~0.60	Trained with full dataset on RTX 3080 Ti
YOLOv8n	0.635	0.487	0.622	0.631	Lightweight; fast but limited
YOLOv8s (final)	0.697	0.541	0.678	0.666	Best performer with augmentations
Notes:
•	The final YOLOv8s model with imgsz=960 and mosaic=1.0 delivered the best balance between precision and recall.
•	Faster R-CNN performance was robust but slightly lower due to slower convergence.
________________________________________
7. Visual Results and Diagnostics
•	Plotted precision, recall, F1 and PR curves
•	Confusion matrices show class confusion; 'others' class benefited most from YOLOv8s
•	Labels correlogram indicated overrepresentation of 'van' class, slightly biasing predictions
________________________________________
8. Conclusions
The iterative refinement of object detection models on UA-DETRAC using YOLOv8 and Faster R-CNN revealed that:
•	YOLOv8s (small) is more effective than YOLOv8n or Faster R-CNN
•	Image resolution, augmentation strategies, and early stopping significantly influence training convergence and final accuracy
•	Faster R-CNN remains a solid anchor-based baseline trained with the full dataset on an RTX 3080 Ti GPU, but YOLOv8 showed superior inference speed and accuracy
________________________________________
9. Future Work
•	Train Faster R-CNN with deeper backbones (e.g., ResNet-101)
•	Integrate ByteTrack or DeepSORT for real-time multi-object tracking
•	Enhance data augmentation with synthetic occlusion and blur
•	Quantize and export models for embedded deployment
•	Extend to nighttime or adverse weather condition detection tasks
________________________________________

