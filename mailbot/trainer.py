from datasets import load_dataset
import math
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from dataloader import AESLCDataset
from typing import Union, Optional

class TrainModel():
    def __init__(self, experiment_run_name: str, model_checkpoint: str = "distilgpt2", output_dir: str = "./model", block_size: Union[str, int] = "default",
                 dataloader_drop_last: bool = True, evaluation_strategy: str = "epoch", save_strategy: str = "epoch",
                 num_train_epochs: int = 10, logging_steps: int = 5, per_device_train_batch_size: int = 4,
                 per_device_eval_batch_size: int = 4, learning_rate: float = 1e-3, lr_scheduler: str = "cosine", warmup_steps: int = 10,
                 gradient_accumulation_steps: int = 4, use_fp16: bool = True, weight_decay: float = 0.05,
                 monitoring_platform: Optional[str] = 'wandb'):
        self.model_checkpoint = model_checkpoint
        self.block_size = block_size
        self.output_dir = output_dir
        self.dataloader_drop_last = dataloader_drop_last
        self.evaluation_strategy = evaluation_strategy
        self.save_strategy = save_strategy
        self.num_train_epochs = num_train_epochs
        self.logging_steps = logging_steps
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.warmup_steps = warmup_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_fp16 = use_fp16
        self.weight_decay = weight_decay
        self.experiment_run_name = experiment_run_name
        self.monitoring_platform = monitoring_platform

    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        return model
    
    def train(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_checkpoint)
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            dataloader_drop_last=self.dataloader_drop_last,
            evaluation_strategy=self.evaluation_strategy,
            save_strategy=self.save_strategy,
            num_train_epochs=self.num_train_epochs,
            logging_steps=self.logging_steps,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            learning_rate=self.learning_rate,
            lr_scheduler_type=self.lr_scheduler,
            warmup_steps=self.warmup_steps,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            fp16=self.use_fp16,
            weight_decay=self.weight_decay,
            run_name=self.experiment_run_name,
            report_to=self.monitoring_platform,
        )
        dataset = AESLCDataset(model_name=self.model_checkpoint, block_size=self.block_size)
        trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        )
        trainer.train()
        self.evaluate_perplexity(trainer)
        return trainer
    
    def evaluate_perplexity(self, trainer):
        eval_results = trainer.evaluate()
        return math.exp(eval_results['eval_loss'])