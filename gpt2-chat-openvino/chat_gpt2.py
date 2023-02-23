from pkg_resources import resource_filename

from transformers import GPT2Tokenizer
from openvino.runtime import Core
import numpy as np

from utils import *

class GPT2Predictor:
    compiled_model = None
    def __init__(self, model_path=resource_filename(__name__,'./model/gpt2.xml')):
        if self.compiled_model is None:
            # initialize openvino core
            core = Core()
            # read the model and corresponding weights from file
            model = core.read_model(model_path)
            # compile the model for CPU devices
            self.compiled_model = core.compile_model(model=model, device_name="CPU")
            # get output tensors
            self.output_key = self.compiled_model.output(0)

            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.eos_token_id = self.tokenizer.eos_token_id

    # this function converts text to tokens
    def tokenize(self, text):
        """
        tokenize input text using GPT2 tokenizer
        Parameters:
        text, str - input text
        Returns:
        input_ids - np.array with input token ids
        attention_mask - np.array with 0 in place, where should be padding and 1 for places where original tokens are located, represents attention mask for model 
        """
        inputs = self.tokenizer(text, return_tensors="np")
        return inputs["input_ids"], inputs["attention_mask"]

    def generate_sequence(self, input_prompt, max_sequence_length=128, dynamic_shapes=True):
        """
        text prediction cycle.

        Parameters:
        input_ids: tokenized input ids for model
        attention_mask: attention mask for model
        max_sequence_length: maximum sequence length for stop iteration
        eos_token_ids: end of sequence index from vocab
        dynamic_shapes: use dynamic shapes for inference or pad model input to max_sequece_length
        Returns:
        predicted token ids sequence
        """

        input_ids, attention_mask = self.tokenize(input_prompt)


        while True:
            cur_input_len = len(input_ids[0])
            if not dynamic_shapes:
                pad_len = max_sequence_length - cur_input_len
                model_input_ids = np.concatenate((input_ids, [[self.eos_token_id] * pad_len]), axis=-1)
                model_input_attention_mask = np.concatenate((attention_mask, [[0] * pad_len]), axis=-1)
            else:
                model_input_ids = input_ids
                model_input_attention_mask = attention_mask
            outputs = self.compiled_model({"input_ids": model_input_ids, "attention_mask": model_input_attention_mask})[self.output_key]
            next_token_logits = outputs[:, cur_input_len - 1, :]
            # pre-process distribution
            next_token_scores = process_logits(cur_input_len,
                                            next_token_logits, self.eos_token_id)
            top_k = 20
            next_token_scores = get_top_k_logits(next_token_scores, top_k)
            # get next token id
            probs = softmax(next_token_scores)
            next_tokens = np.random.choice(probs.shape[-1], 1,
                                        p=probs[0], replace=True)
            # break the loop if max length or end of text token is reached
            if cur_input_len == max_sequence_length or next_tokens == self.eos_token_id:
                break
            else:
                input_ids = np.concatenate((input_ids, [next_tokens]), axis=-1)
                attention_mask = np.concatenate((attention_mask, [[1] * len(next_tokens)]), axis=-1)

        output_ids = input_ids

        output_text = " "
        # Convert IDs to words and make the sentence from it
        for i in output_ids[0]:
            output_text += self.tokenizer.convert_tokens_to_string(self.tokenizer._convert_id_to_token(i))

        return output_text