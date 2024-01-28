# %%
train_dataset='/home/recallfun/webapps/foodblog/fine-tune-llm/alpaca_finetune_dataset.jsonl'

# %%
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from trl import SFTTrainer

# %%
train_dataset = load_dataset('json', data_files=train_dataset , split='train')

# %%
import matplotlib.pyplot as plt

def plot_data_lengths(tokenized_train_dataset):
    lengths = [len(x['input_ids']) for x in tokenized_train_dataset]
    print(len(lengths))
    print(max(lengths))

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=20, alpha=0.7, color='blue')
    plt.xlabel('Length of input_ids')
    plt.ylabel('Frequency')
    plt.title('Distribution of Lengths of input_ids')
    plt.show()

# %%

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import Accelerator

current_device = Accelerator().process_index

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_name = "/home/recallfun/llm/models/mistral-7b"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'
train_dataset = train_dataset.map(lambda samples: tokenizer(samples["text"]), batched=True)
model = AutoModelForCausalLM.from_pretrained(model_name,
  device_map="auto",
  quantization_config=bnb_config)
plot_data_lengths(train_dataset)

# %%
from peft import prepare_model_for_kbit_training

train_dataset = train_dataset.map(lambda samples: tokenizer(samples["text"],truncation=True,
        max_length=2000,
        padding="max_length",), batched=True)
plot_data_lengths(train_dataset)
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# %%
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

# %%
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8, 
    lora_alpha=16, 
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "w1",
        "w2",
        "w3",
        "lm_head",
    ],
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)

peft_model = get_peft_model(model, config)
print_trainable_parameters(peft_model)

# %%
from transformers import TrainingArguments

training_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        learning_rate=2.5e-5,
        fp16=True,
        logging_steps=25,
        output_dir="/home/recallfun/llm/finetune/food-blog-mistral-7b-checkpoints-alpaca",
        max_steps=500,
        save_total_limit = 2,
save_steps = 250,
load_best_model_at_end=False,
num_train_epochs=3,
optim="paged_adamw_8bit"
    )

# %%
tokenizer.pad_token = tokenizer.eos_token
trainer = SFTTrainer(
    model=peft_model,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=2000,  # You can specify the maximum sequence length here
    tokenizer=tokenizer,
    args=training_args,
    packing=False,
)
peft_model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

# Start the training process
trainer.train()

# Save the fine-tuned model
trainer.model.save_pretrained("/home/recallfun/llm/finetune/food-blog-mistral-7b-finetune-alpaca")


