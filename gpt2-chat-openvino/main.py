# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# from pathlib import Path
# from openvino.runtime import serialize
# from openvino.tools import mo
# from transformers.onnx import export, FeaturesManager

from chat_gpt2 import GPT2Predictor
from utils import *

# def download_and_convert_to_openvino_ir():

#     # paper: https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
#     # HuggingFace model: https://huggingface.co/gpt2

#     tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#     pt_model = GPT2LMHeadModel.from_pretrained('gpt2')

#     ##############################
#     # Convert GPT-2 to OpenVINO IR
#     ##############################

#     ###### Convert from Pytorch ->  ONNX -> OpenVINO IR

#     # define path for saving onnx model
#     onnx_path = Path("model/gpt2.onnx")
#     onnx_path.parent.mkdir(exist_ok=True)

#     # define path for saving openvino model
#     model_path = onnx_path.with_suffix(".xml")

#     # get model onnx config function for output feature format casual-lm
#     model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(pt_model, feature='causal-lm')

#     # fill onnx config based on pytorch model config
#     onnx_config = model_onnx_config(pt_model.config)

#     # convert model to onnx
#     onnx_inputs, onnx_outputs = export(tokenizer, pt_model, onnx_config, onnx_config.default_onnx_opset, onnx_path)

#     # convert model to openvino
#     ov_model = mo.convert_model(onnx_path, compress_to_fp16=True, input="input_ids[1,1..128],attention_mask[1,1..128]")

#     # serialize openvino model
#     serialize(ov_model, str(model_path))
#     return model_path, tokenizer

def main():
    ########################################################
    # Download the model from HF and convert to OpenVINO IR
    ########################################################
    # model_path, tokenizer = download_and_convert_to_openvino_ir()

    chatgpt2 = GPT2Predictor()

    while True:
        text = input('\n[YOU]:')
        if text == 'quit':
            print('[INFO] EXITING...')
            break
        response = chatgpt2.generate_sequence(text)
        print('\n[ChatGPT2]:', response.replace('\t','\n'))

if __name__ == '__main__':
    main()