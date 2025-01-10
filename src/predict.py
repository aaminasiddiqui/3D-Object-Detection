import argparse
import os
import cv2
import torch
import numpy as np
from easydict import EasyDict as edict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from data_process import kitti_data_utils, kitti_bev_utils
from models.model_utils import create_model
from utils.evaluation_utils import post_processing_v2, rescale_boxes
from utils.visualization_utils import show_image_with_boxes, merge_rgb_to_bev, predictions_to_kitti_format
from utils.misc import time_synchronized


def parse_test_configs():
    parser = argparse.ArgumentParser(description='Single image 3D Bounding Box Generator')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--calib_path', type=str, required=True, help='Path to the calibration file')
    parser.add_argument('-a', '--arch', type=str, default='darknet', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--use_giou_loss', action='store_true',
                        help='If true, use GIoU loss during training. If false, use MSE loss for training')
    parser.add_argument('--saved_fn', type=str, default='complexer_yolov4', help='Model save name')
    parser.add_argument('--cfgfile', type=str, default='./config/cfg/complex_yolov4.cfg', help='Path to the cfg file')
    parser.add_argument('--pretrained_path', type=str, default='./saved_model/Model_complexer_yolo_epoch_25.pth',
                        help='Path to the pretrained model')
    parser.add_argument('--conf_thresh', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--nms_thresh', type=float, default=0.5, help='NMS threshold')
    parser.add_argument('--img_size', type=int, default=608, help='Input image size')

    configs = edict(vars(parser.parse_args()))
    configs.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return configs


def generate_3d_bounding_boxes(configs):
    # Load the model
    model = create_model(configs)
    model.load_state_dict(torch.load(configs.pretrained_path, map_location=configs.device))
    model = model.to(configs.device)
    model.eval()

    # Prepare the image
    img_rgb = cv2.imread(configs.image_path)
    if img_rgb is None:
        raise FileNotFoundError(f"Image not found at {configs.image_path}")

    calib = kitti_data_utils.Calibration(configs.calib_path)
    img_bev = kitti_bev_utils.create_bev_from_rgb(img_rgb, calib)
    img_bev = cv2.resize(img_bev, (configs.img_size, configs.img_size))
    img_tensor = torch.from_numpy(img_bev).unsqueeze(0).to(configs.device).float()

    # Run the model
    t1 = time_synchronized()
    with torch.no_grad():
        outputs = model(img_tensor)
    t2 = time_synchronized()

    # Post-processing
    detections = post_processing_v2(outputs, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh)

    if detections[0] is None:
        print("No objects detected.")
        return

    detections = rescale_boxes(detections[0], configs.img_size, img_rgb.shape[:2])

    # Visualization
    for x, y, w, l, im, re, *_, cls_pred in detections:
        yaw = np.arctan2(im, re)
        kitti_bev_utils.drawRotatedBox(img_bev, x, y, w, l, yaw, kitti_bev_utils.colors[int(cls_pred)])

    objects_pred = predictions_to_kitti_format([detections], calib, img_rgb.shape, configs.img_size)
    img_with_boxes = show_image_with_boxes(img_rgb, objects_pred, calib, False)

    output_path = os.path.join('./results/', os.path.basename(configs.image_path))
    os.makedirs('./results/', exist_ok=True)
    cv2.imwrite(output_path, img_with_boxes)

    print(f"Done processing {configs.image_path}, time: {(t2 - t1) * 1000:.1f}ms")
    print(f"Output saved at {output_path}")


if __name__ == '__main__':
    configs = parse_test_configs()
    generate_3d_bounding_boxes(configs)
