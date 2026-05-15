
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_model(num_classes=4):
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


# mapping class (ID 0 = background, di-remap dari dataset COCO: 0→1, 1→2, 2→3)
CLASS_NAMES = {
    0: 'background',
    1: 'Trash',
    2: 'plastic_bag',
    3: 'plastic_wrapper',
}

CLASS_COLORS = {
    1: (217,  83,  25),   # Trash           = merah-oranye
    2: (0,   114, 189),   # plastic_bag     = biru
    3: (50,  205,  50),   # plastic_wrapper = hijau
}
