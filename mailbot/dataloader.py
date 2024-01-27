from datasets import load_dataset
from transformers import AutoTokenizer

# dataset = load_dataset("postbot/aeslc_kw")

class AESLCDataset():
    def __init__(self, model_name, block_size = 1024):
        self.dataset_name = "postbot/aeslc_kw"
        self.model_name = model_name
        self.block_size = block_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    def load_data(self):
        dataset = load_dataset(self.dataset_name)
        return dataset
    
    def group_texts(self, examples):
        if self.block_size == 'default':
            self.block_size = self.tokenizer.model_max_length
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // self.block_size) * self.block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + self.block_size] for i in range(0, total_length, self.block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def tokenize_function(self, examples):
        return self.tokenizer(examples["clean_email"])

    def get_tokenized_dataset(self, dataset):
        tokenized_datasets = dataset.map(self.tokenize_function, batched=True, num_proc=4,
                                 remove_columns=['email_body', 'subject_line',
                                                 'clean_email', 'clean_email_keywords'])
        return tokenized_datasets
    
    def tokenize_and_group_dataset(self):
        dataset = self.load_data()
        tokenized_datasets = self.get_tokenized_dataset(dataset)
        lm_datasets = tokenized_datasets.map(
        self.group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
        )
        return lm_datasets



