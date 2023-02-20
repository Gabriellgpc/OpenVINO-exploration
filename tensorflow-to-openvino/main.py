from openvino.runtime import Core
import tensorflow as tf
import numpy as np
import cv2

from pkg_resources import resource_filename
from pathlib import Path
import time
import os

if __name__ == '__main__':
    image_path = resource_filename(__name__, '../data/images/burguer.jpg')
    model_path = Path(resource_filename(__name__, './model/mobilenetv3_small_224_1.0_float'))
    openvino_ir_path = Path(resource_filename(__name__, './model/mobilenetv3_small_224_1.0_float.xml'))

    print('[INFO] Download model....')
    model = tf.keras.applications.MobileNetV3Small()
    print('[INFO] Saving TF model to ', model_path)
    model.save(model_path)

    """
    ## Convert a Model to OpenVINO IR Format
    ### Convert a TensorFlow Model to OpenVINO IR Format
    Use Model Optimizer to convert a TensorFlow model to OpenVINO IR with `FP16` precision.
    The models are saved to the current directory. Add mean values to the model and scale
    the output with the standard deviation with `--scale_values`. With these options,
    it is not necessary to normalize input data before propagating it through the network.
    The original model expects input images in `RGB` format. The converted model also expects images in `RGB` format.
    If you want the converted model to work with `BGR` images, use the `--reverse-input-channels` option.
    First construct the command for Model Optimizer,
    and then execute this command in the notebook by prepending the command with an `!`.
    There may be some errors or warnings in the output. When model optimization is successful,
    the last lines of the output will include `[ SUCCESS ] Generated IR version 11 model.`

    For more information about Model Optimizer, including a description of the command-line options,
    see the [Model Optimizer Developer Guide]
    (https://docs.openvino.ai/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).
    For information about the model, including input shape, expected color order and mean values,
    refer to the [model documentation](https://docs.openvino.ai/latest/omz_models_model_mobilenet_v3_small_1_0_224_tf.html).
    """

    # Construct the command for Model Optimizer.
    mo_command = f"""mo
                    --saved_model_dir "{model_path}"
                    --input_shape "[1,224,224,3]"
                    --model_name "{model_path.name}"
                    --compress_to_fp16
                    --output_dir "{model_path.parent}"
                    """
    mo_command = " ".join(mo_command.split())
    print("[INFO] Model Optimizer command to convert TensorFlow to OpenVINO:")
    print(mo_command)
    os.system(mo_command)


    # Testing
    ie = Core()
    model = ie.read_model(openvino_ir_path)
    compiled_model = ie.compile_model(model=model, device_name="CPU")

    IMAGENET_CLASS_LIST = resource_filename( __name__, '../data/imagenet_2012.txt')
    INPUT_SIZE = [224,224]

    # load the input image and preprocess it
    image = cv2.imread( image_path )
    # input_image will be feeding into the model
    input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(input_image, dsize=INPUT_SIZE)
    input_image = np.expand_dims(input_image, axis=0)
    #[1,224,224,3]. format [N, H, W, C]

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
    imagenet_classes = imagenet_classes

    print('[INFO] PREDICTED INDEX:', result_index)
    print('[INFO] PREDICTED CLASS:', imagenet_classes[result_index])

    viz_image  = cv2.resize(image, dsize=INPUT_SIZE)
    cv2.imshow('input',viz_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
