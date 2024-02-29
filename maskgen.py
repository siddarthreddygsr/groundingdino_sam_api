import supervision as sv
import torch
from groundingdino.util.inference import Model
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import warnings
from typing import List
import cv2
import numpy as np
import matplotlib.pyplot as plt
from generate_filename import randomizer
from PIL import Image
from np2b64 import convert_to_url
from fooocus import focus_endpoint

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GROUNDING_DINO_CHECKPOINT_PATH = "./weights/groundingdino_swint_ogc.pth"
GROUNDING_DINO_CONFIG_PATH = "./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
model_type = "vit_t"
sam_checkpoint = "./MobileSAM/weights/mobile_sam.pt"
device = "cuda"
mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam.to(device=device)
mobile_sam.eval()
sam_predictor = SamPredictor(mobile_sam)
CLASSES = ['lips']
BOX_TRESHOLD = 0.40
TEXT_TRESHOLD = 0.25

def enhance_class_name(class_names: List[str]) -> List[str]:
    return [
        f"all {class_name}s"
        for class_name
        in class_names
    ]
def large_lip_logic(image_path,pod_id):
    image = cv2.imread(image_path)

    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=enhance_class_name(class_names=CLASSES),
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )

    box_annotator = sv.BoxAnnotator()
    box_annotator = sv.BoxAnnotator()
    labels = [
        f"{CLASSES[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, _
        in detections]
    annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)
    sv.plot_image(annotated_frame, (8, 8))
    for x1, y1, x2, y2 in detections.xyxy:
        x1 = x1 + 5
        x2 = x2 + 5
        y1 = y1 + 5
        y2 = y2 + 5
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        mask = np.zeros_like(image, dtype=np.uint8)
        cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), thickness=cv2.FILLED)
    file_path = f"processed_image/{randomizer()}.png"
    cv2.imwrite(file_path,mask)
    mask_url = convert_to_url(file_path)
    image_url = convert_to_url(image_path)
    b64 = focus_endpoint(image_url,mask_url,pod_id)
    return b64
    # return file_path

def medium_lip_logic(image_path,pod_id):
    image = cv2.imread(image_path)
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=enhance_class_name(class_names=CLASSES),
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )

    box_annotator = sv.BoxAnnotator()
    labels = [
        f"{CLASSES[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, _
        in detections]
    annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

    sv.plot_image(annotated_frame, (8, 8))
    for x1, y1, x2, y2 in detections.xyxy:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        mask = np.zeros_like(image, dtype=np.uint8)
        cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), thickness=cv2.FILLED)

    file_path = f"processed_image/{randomizer()}.png"
    cv2.imwrite(file_path,mask)
    mask_url = convert_to_url(file_path)
    image_url = convert_to_url(image_path)
    b64 = focus_endpoint(image_url,mask_url,pod_id)
    return b64
    # return file_path

def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)

def light_lip_logic(image_path,pod_id):
    try:
        CLASSES = ['lip']
        BOX_TRESHOLD = 0.40
        TEXT_TRESHOLD = 0.25

        image = cv2.imread(image_path)
        # detect objects
        detections = grounding_dino_model.predict_with_classes(
            image=image,
            classes=enhance_class_name(class_names=CLASSES),
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )

        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        labels = [
            f"{CLASSES[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _
            in detections]
        annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)
        detections.mask = segment(
            sam_predictor=sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )

        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        labels = [
            f"{CLASSES[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _
            in detections]
        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_image1 = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

        # cv2.imwrite('ha_mask.png',annotated_image)
        sv.plot_image(annotated_image1, (8, 8))


        mask_pil = Image.fromarray(detections.mask[0])
        file_path = f"processed_image/{randomizer()}.png"
        mask_pil.save(file_path)
        mask_url = convert_to_url(file_path)
        image_url = convert_to_url(image_path)
        b64 = focus_endpoint(image_url,mask_url,pod_id)
        return b64
        # return file_path
    except:
        return {"error" : "lips not detected"}

