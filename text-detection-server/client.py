from utils import convert_result_to_image


import cv2
import numpy as np
import matplotlib.pyplot as plt
from ovmsclient import make_grpc_client


def main():
    image_file = '../data/images/intel_rnb.jpg'
    address = "localhost:9000"
    model_name = 'text-detection'

    # Bind the grpc address to the client object
    client = make_grpc_client(address)

    # request model status
    model_status = client.get_model_status(model_name=model_name)
    print('[INFO] MODEL STATUS', model_status)

    # request model metadata
    model_metadata = client.get_model_metadata(model_name=model_name)
    print('[INFO] MODEL METADATA', model_metadata)

    # load input image
    image = cv2.imread(image_file)
    fp_image = image.astype("float32")

    # Resize the image to meet network expected input sizes.
    input_shape = model_metadata['inputs']['image']['shape']
    height, width = input_shape[2], input_shape[3]
    resized_image = cv2.resize(fp_image, (height, width))

    # Reshape to the network input shape.
    input_image = np.expand_dims(resized_image.transpose(2, 0, 1), 0)
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # cv2.imshow('input')

    # request prediction on a numpy array
    inputs = {"image": input_image}

    # Run inference on model server and receive the result data
    boxes = client.predict(inputs=inputs, model_name=model_name)['boxes']

    # Remove zero only boxes.
    boxes = boxes[~np.all(boxes == 0, axis=1)]
    print(boxes)

    # plt.figure(figsize=(10, 6))
    # plt.axis("off")
    # plt.imshow(convert_result_to_image(image, resized_image, boxes, conf_labels=False))
    viz_image = convert_result_to_image(image, resized_image, boxes, conf_labels=False)[:,:,::-1]
    cv2.imshow('out', viz_image)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()