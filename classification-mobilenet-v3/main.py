import cv2
import numpy as np
from openvino.runtime import Core
from pkg_resources import resource_filename
import time

import click

@click.command()
@click.option('--model', default=resource_filename(__name__, './model/v3-small_224_1.0_float.xml'))
@click.option('--image-path', '--image', default=resource_filename(__name__, './images/burguer.jpg'))
def main(model, image_path):
    IMAGENET_CLASS_LIST = resource_filename( __name__, './imagenet_2012.txt')
    INPUT_SIZE = [224,224]

    # load the input image and preprocess it
    image = cv2.imread( image_path )
    # input_image will be feeding into the model
    input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(input_image, dsize=INPUT_SIZE)
    input_image = np.expand_dims(input_image, axis=0)
    #[1,224,224,3]. format [N, H, W, C]

    # load and compile the model
    ie = Core()

    start_time = time.process_time()
    model = ie.read_model(model=model)
    compiled_model = ie.compile_model(model=model, device_name='CPU')
    end_time = time.process_time()

    print('[INFO] Load and compile model time: {:.4f} ms'.format((end_time - start_time)*1000.0))

    input_layer  = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    print(f"[INFO] input: {input_layer.any_name}")
    print(f"[INFO] input precision: {input_layer.element_type}")
    print(f"[INFO] input shape: {input_layer.shape}")

    print(f"[INFO] output: {output_layer.any_name}")
    print(f"[INFO] output precision: {output_layer.element_type}")
    print(f"[INFO] output shape: {output_layer.shape}")

    # do inference
    start_time = time.process_time()
    result_infer = compiled_model([input_image])[output_layer]
    result_index = np.argmax(result_infer)
    end_time = time.process_time()

    print('[INFO] inference time: {:.4f} ms'.format((end_time - start_time)*1000.0))

    # Convert the inference result to a class name.
    imagenet_classes = open(IMAGENET_CLASS_LIST).read().splitlines()
    # The model description states that for this model, class 0 is a background.
    # Therefore, a background must be added at the beginning of imagenet_classes.
    imagenet_classes = ['background'] + imagenet_classes

    print('[INFO] PREDICTED INDEX:', result_index)
    print('[INFO] PREDICTED CLASS:', imagenet_classes[result_index])

    viz_image  = cv2.resize(image, dsize=INPUT_SIZE)
    cv2.imshow('input',viz_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()