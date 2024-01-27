# MailCompletion-bot

A simple large language AI bot to assist in writing emails. It utilizes a finetuned version of distilgpt2 on an extensively cleaned email dataset and generates output in causal language modeling (CLM) approach, that is, generates the next words based from the previous words.

## AESLC Dataset
The dataset we have utilized is a modified and cleaned version of the AESLC dataset. The original dataset is introduced in the paper [**This Email Could Save Your Life: Introducing the Task of Email Subject Line Generation**](https://arxiv.org/pdf/1906.03497v1.pdf) and is used to generate a subject line from the email body. We utilize a clean version of the dataset that removes the private data like emails, phone number and other factual information by masking them in a named entity recognition fashion. For more detailed analysis, check out https://aeslc-kw-train-eda.netlify.app/.

Thanks to team postbot for cleaning the dataset. Find the dataset at [HuggingFace](https://huggingface.co/datasets/postbot/aeslc_kw).

## distilgpt2 Model
The language model fine-tuned is Distilled-GPT2 model which is an English-language model pre-trained with the supervision of the 124M parameters version of Generative Pre-trained Transformer 2 (GPT-2). DistilGPT2, which has 82M parameters, was developed using knowledge distillation and was designed to be a faster, lighter version of GPT-2.

## Finetuning process
Model and Tokenizer Loading:
- Loads the DistilGPT2 model and its fast tokenizer.

Data Preparation:
- Tokenizes the "clean_email" column of the dataset.
- Efficiently groups texts into blocks for model compatibility, dropping small remainders.
- Creates labels from input IDs for language modeling.

Training Configuration:
- Specifies output directory, evaluation, and saving strategies, number of epochs, logging frequency, batch sizes, learning rate, scheduler, warmup steps, gradient accumulation, mixed-precision (fp16), weight decay, run name, and reporting to Weights & Biases.

T4-Specific Optimizations:
- Uses fp16 for accelerated half-precision training, well-suited to T4's Tensor Cores.
- Leverages multi-processing for data preparation to maximize CPU utilization, complementing GPU-accelerated training.

Training:
- Initiates the training process using the specified model, arguments, datasets, and T4 GPU.
- Trained for 10 epochs with a learning rate of 1e-3 which took around 40 mins.

## Performance
The model achieves 33.20 perplexity score on the AESLC dataset.

## Conclusion
The model can be used for autocompletion in the case of writing emails thus serving its purpose. But for better accuracy, it is recommended to not generate more than 5 next-token predictions to reduce out-of-context output generation.

## Citations
```
@inproceedings{sanh2019distilbert,
  title={DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter},
  author={Sanh, Victor and Debut, Lysandre and Chaumond, Julien and Wolf, Thomas},
  booktitle={NeurIPS EMC^2 Workshop},
  year={2019}
}
@InProceedings{zhang2019slg,
  author =      "Rui Zhang and Joel Tetreault",
  title =       "This Email Could Save Your Life: Introducing the Task of Email Subject Line Generation",
  booktitle =   "Proceedings of The 57th Annual Meeting of the Association for Computational Linguistics",
  year =        "2019",
  address =     "Florence, Italy"
}
```
