import numpy as np
import cv2


CLASS_NAMES = {
    0: "Fall-Detected",
    1: "Gloves",
    2: "Goggles",
    3: "Hardhat",
    4: "Ladder",
    5: "Mask",
    6: "NO-Gloves",
    7: "NO-Goggles",
    8: "NO-Hardhat",
    9: "NO-Mask",
    10: "NO-Safety Vest",
    11: "Person",
    12: "Safety Cone",
    13: "Safety Vest",
}


def postprocess(
    output,
    original_width,
    original_height,
    confidence_threshold=0.25,
    nms_threshold=0.45
):

    predictions = output[0]
    predictions = predictions.T
    boxes = []
    scores = []
    class_ids = []

    for prediction in predictions:
        cx, cy, w, h = prediction[:4]
        class_scores = prediction[4:]
        class_id = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id])
        if confidence < confidence_threshold:
            continue

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        scale_x = original_width / 640
        scale_y = original_height / 640

        x1 *= scale_x
        y1 *= scale_y
        x2 *= scale_x
        y2 *= scale_y

        x1 = max(0, min(original_width - 1, x1))
        y1 = max(0, min(original_height - 1, y1))
        x2 = max(0, min(original_width - 1, x2))
        y2 = max(0, min(original_height - 1, y2))

        box_width = x2 - x1
        box_height = y2 - y1

        boxes.append([
            int(x1),
            int(y1),
            int(box_width),
            int(box_height)
        ])

        scores.append(confidence)
        class_ids.append(class_id)
    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        confidence_threshold,
        nms_threshold
    )

    detections = []

    if len(indices) > 0:

        for i in indices:
            i = int(i)
            detection = {
                "class_id": class_ids[i],
                "class_name": CLASS_NAMES[class_ids[i]],
                "confidence": scores[i],
                "box": boxes[i]
            }

            detections.append(detection)

    return detections