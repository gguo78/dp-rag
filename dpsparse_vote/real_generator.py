# real_generator.py

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class RealGenerator:
    def __init__(self, model_name='facebook/opt-1.3b', device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = self.model.to(self.device)

    def generate_next_token(self, prompt, max_new_tokens=1):
        # Tokenize the input prompt
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

        # Generate next token
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Deterministic greedy decoding
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Get the newly generated token
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        generated_token = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        return generated_token if generated_token else '<EOS>'
