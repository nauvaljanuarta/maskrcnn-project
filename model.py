# model.py
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_model(num_classes=7):
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)

    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_channels=in_features_mask,
        dim_reduced=256,
        num_classes=num_classes
    )

    return model


# mapping class
CLASS_NAMES = {
    0: 'background',
    1: 'Cloth',
    2: 'Foam',
    3: 'Hard Plastic',
    4: 'Other',
    5: 'Paper',
    6: 'Soft Plastic',
}

CLASS_COLORS = {
    1: (0,   114, 189),   # Cloth        = biru
    2: (50,  205,  50),   # Foam         = hijau
    3: (217,  83,  25),   # Hard Plastic = merah-oranye
    4: (237, 177,  32),   # Other        = kuning
    5: (126,  47, 142),   # Paper        = ungu
    6: (255, 140,   0),   # Soft Plastic = oren
}
