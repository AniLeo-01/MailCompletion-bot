from datasets import load_dataset
import math
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from dataloader import AESLCDataset
from typing import Union, Optional
import argparse

class TrainModel():
    def __init__(self, experiment_run_name: str, save_current_model_path: Optional[str], output_dir: str,
                 model_checkpoint: str = "distilgpt2", block_size: Union[str, int] = "default",
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
        self.save_current_model_path = save_current_model_path

    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        return model
    
    def train_model(self):
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
        perplexity = self.evaluate_perplexity(trainer)
        print(f"Perplexity: {perplexity:.2f}")
        if self.save_current_model_path:
            trainer.save_model(self.save_current_model_path)
        return trainer
    
    def evaluate_perplexity(self, trainer):
        eval_results = trainer.evaluate()
        return math.exp(eval_results['eval_loss'])
    
if __name__ == "__main__":
    #add cli args
    ap = argparse.ArgumentParser()
    ap.add_argument('-r', '--run_name', required=True, help='experiment run name')
    ap.add_argument('--save_model', required=True, help='save current model path')
    ap.add_argument('-o', '--output_dir', required=True, help='output directory')
    ap.add_argument('-m', '--model_checkpoint', required=False, help='model checkpoint')
    ap.add_argument('-b', '--block_size', required=False, help='block size')
    ap.add_argument('-d', '--dataloader_drop_last', required=False, help='dataloader drop last')
    ap.add_argument('-e', '--evaluation_strategy', required=False, help='evaluation strategy')
    ap.add_argument('-s', '--save_strategy', required=False, help='save strategy')
    ap.add_argument('-n', '--num_train_epochs', required=False, help='num train epochs')
    ap.add_argument('-l', '--logging_steps', required=False, help='logging steps')
    ap.add_argument('-p', '--per_device_train_batch_size', required=False, help='per device train batch size')
    ap.add_argument('-q', '--per_device_eval_batch_size', required=False, help='per device eval batch size')
    ap.add_argument('-l', '--learning_rate', required=False, help='learning rate')
    ap.add_argument('-c', '--lr_scheduler', required=False, help='lr scheduler')
    ap.add_argument('-w', '--warmup_steps', required=False, help='warmup steps')
    ap.add_argument('-g', '--gradient_accumulation_steps', required=False, help='gradient accumulation steps')
    ap.add_argument('-f', '--use_fp16', required=False, help='use fp16')
    ap.add_argument('-w', '--weight_decay', required=False, help='weight decay')
    ap.add_argument('-w', '--monitoring_platform', required=True, help='monitoring platform')

    args = vars(ap.parse_args())
    trainer = TrainModel(**args)
    trainer.train_model()