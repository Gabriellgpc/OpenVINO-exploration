from openvino.runtime import Core
import numpy as np
from pkg_resources import resource_filename
import cv2

from pathlib import Path

import numpy as np
import cv2

def non_max_suppression(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the boxes by
    # their bottom-right y-coordinate
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index
        # value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the
        # bounding box and the smallest (x, y) coordinates for the
        # end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap between the bounding box and
        # other bounding boxes
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap
        # greater than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked
    return boxes[pick]


def main():
    # model_path = Path( __name__, 'model/public/' )
    model_ir_path = Path(resource_filename(__name__, 'model/public/ultra-lightweight-face-detection-slim-320/FP16/ultra-lightweight-face-detection-slim-320.xml'))
    image_path = Path(resource_filename(__name__, '../data/images/people.jpg'))

    # load and compile the model
    ie = Core()
    model = ie.read_model(model=model_ir_path)
    compiled_model = ie.compile_model(model=model, device_name='CPU')

    # read the image and preprocess it to fit the model input expected
    print('[DEBUG] Image path',image_path.absolute())
    image = cv2.imread(str(image_path))
    input_image = cv2.resize(image, dsize=[320,240])
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = np.expand_dims(input_image.transpose(2,0,1), axis=0)

    input_layer  = compiled_model.input(0)
    output_scores_layer = compiled_model.output(0)
    output_boxes_layer  = compiled_model.output(1)

    print(f"[INFO] input_image: {input_image.shape}")

    print(f"[INFO] input name: {input_layer.any_name}")
    print(f"[INFO] input precision: {input_layer.element_type}")
    print(f"[INFO] input shape: {input_layer.shape}")

    print(f"[INFO] output name: {output_scores_layer.any_name}")
    print(f"[INFO] output precision: {output_scores_layer.element_type}")
    print(f"[INFO] output shape: {output_scores_layer.shape}")

    print(f"[INFO] output name: {output_boxes_layer.any_name}")
    print(f"[INFO] output precision: {output_boxes_layer.element_type}")
    print(f"[INFO] output shape: {output_boxes_layer.shape}")


    # inference
    scores_pred = compiled_model( [input_image] )[output_scores_layer]
    bboxes_pred = compiled_model( [input_image] )[output_boxes_layer]
    print(scores_pred)
    print(scores_pred.shape)

    print(bboxes_pred)
    print(bboxes_pred.shape)

    # get all predictions with more than 0.5 of confidence
    filtered_indexes = np.argwhere( scores_pred[0,:,1] > 0.6  ).tolist()
    filtered_boxes   = bboxes_pred[0,filtered_indexes,:]
    filtered_scores  = scores_pred[0,filtered_indexes,1]
    # print('[DEBUG] filtered_indexes',filtered_indexes)
    # print('[DEBUG] filtered_scores', scores_pred[0,filtered_indexes,:])
    print('[INFO] confidences', filtered_scores.tolist())
    print('[INFO] filtered_boxes', filtered_boxes)
    print('[INFO] #Faces detected', len(filtered_boxes))

    # For each detection, the description has the format: [x_min, y_min, x_max, y_max], where:
    # (x_min, y_min) - coordinates of the top left bounding box corner (coordinates are in normalized format, in range [0, 1])
    # (x_max, y_max) - coordinates of the bottom right bounding box corner (coordinates are in normalized format, in range [0, 1])
    h, w = image.shape[:2]


    # convert all boxes to image coordinates
    def _convert_bbox_format(*args):
        bbox = args[0]
        x_min, y_min, x_max, y_max = bbox
        x_min = int(w*x_min)
        y_min = int(h*y_min)
        x_max = int(w*x_max)
        y_max = int(h*y_max)
        return x_min, y_min, x_max, y_max

    print('filtered_boxes.shape', filtered_boxes.shape)
    bboxes_image_coord = np.apply_along_axis(_convert_bbox_format, axis = 2, arr=filtered_boxes)

    # apply non-maximum supressions
    bboxes_image_coord = non_max_suppression(bboxes_image_coord.reshape([-1,4]), overlapThresh=0.5)
    print('filtered_boxes.shape', bboxes_image_coord.shape)

    # draw all bboxes on the input image
    for boxe in bboxes_image_coord:
        x_min, y_min, x_max, y_max = boxe

        pt1 = (x_min, y_min)
        pt2 = (x_max, y_max)
        cv2.rectangle(image, pt1, pt2, color=[0, 255, 0], thickness=2, lineType=cv2.LINE_4)#BGR

    cv2.imshow('input', image)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()