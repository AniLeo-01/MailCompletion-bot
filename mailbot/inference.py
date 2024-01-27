from transformers import AutoModelForCausalLM, AutoTokenizer

class InferenceModel():
    def __init__(self, model_checkpoint: str = "distilgpt2", return_tensors='pt',
                 return_dict_in_generate=True, num_beams=1, 
                 do_sample=False, repetition_penalty=1.5, length_penalty=2.0):
        self.model_checkpoint = model_checkpoint
        self.return_tensors = return_tensors
        self.return_dict_in_generate = return_dict_in_generate
        self.num_beams = num_beams
        self.do_sample = do_sample
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty

    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_checkpoint)
        return model
    
    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        return tokenizer
    
    def generate_text(self, input_text, num_token_generate=1):
        model = self.load_model()
        tokenizer = self.load_tokenizer()
        inputs = tokenizer(input_text, return_tensors=self.return_tensors)
        generation_output = model.generate(
            **inputs,
            return_dict_in_generate=self.return_dict_in_generate,
            output_scores=self.output_scores,
            max_length=inputs.input_ids.shape[-1]+num_token_generate,  # Limit generation to num_token_generate
            # no_repeat_ngram_size=2,  # Avoid repeating word pairs
            num_beams=self.num_beams,
            do_sample=self.do_sample,  # Use greedy search for deterministic single-word output
            repetition_penalty=self.repetition_penalty,
            length_penalty=self.length_penalty
        )

        generated_word = tokenizer.decode(generation_output['sequences'][0])
        return generated_word