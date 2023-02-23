# GPT-2 Text Prediction with OpenVINO

This notebook shows a text prediction with OpenVINO. We use the [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) model, which is a part of the Generative Pre-trained Transformer (GPT) family. GPT-2 is pre-trained on a large corpus of English text using unsupervised training. The model is available from [HuggingFace](https://huggingface.co/gpt2). GPT-2 displays a broad set of capabilities, including the ability to generate conditional synthetic text samples of unprecedented quality, where we prime the model with an input and have it generate a lengthy continuation.

The following image illustrates complete demo pipeline used for this scenario:

![image2](https://user-images.githubusercontent.com/91228207/163990722-d2713ede-921e-4594-8b00-8b5c1a4d73b5.jpeg)

Model input is tokenized text, which serves as initial condition for generation, then logits from model inference result should be obtained and token with the highest probability is selected using top-k sampling strategy and joined to input sequence. The procedure repeats until end of sequence token will be recived or specified maximul length will be reached. After that, decoding token ids to text using tokenized should be applied.

## Convert GPT-2 to OpenVINO IR

![conversion_pipeline](https://user-images.githubusercontent.com/29454499/211261803-784d4791-15cb-4aea-8795-0969dfbb8291.png)

For starting work with GPT2 model using OpenVINO, model should be converted to OpenVINO Intermediate Represenation (IR) format. HuggingFace provided gpt2 model is PyTorch model, which supported in OpenVINO via conversion to ONNX. We will use HuggingFace transformers library capabilities for export model to ONNX. `transformers.onnx.export` accept preprocessing function for input sample generation (tokenizer in our case), instance of model, ONNX export configuration, ONNX opset version for export and output path. More information about transformers export to ONNX can be found in HuggingFace [documentation](https://huggingface.co/docs/transformers/serialization).

While ONNX models are directly supported by OpenVINO runtime, it can be useful to convert them to IR format to take advantage of OpenVINO optimization tools and features.
`mo.convert_model` python function can be used for converting model using [OpenVINO Model Optimizer](https://docs.openvino.ai/latest/openvino_docs_MO_DG_Python_API.html). The function returns instance of OpenVINO Model class, which is ready to use in Python interface but can also be serialized to OpenVINO IR format for future execution using `openvino.runtime.serialize`. In our case, `compress_to_fp16` parameter is enabled for compression model weights to fp16 precision and also specified dynamic input shapes with possible shape range (from 1 token to maximum length defined in our processing function) for optimization of memory consumption.

# Reference

[OpenVINO notebook example](https://github.com/openvinotoolkit/openvino_notebooks)