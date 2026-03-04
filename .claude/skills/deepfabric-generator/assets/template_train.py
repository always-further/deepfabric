from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# Load the generated dataset
dataset = load_dataset("json", data_files="dataset.jsonl", split="train")

config = SFTConfig(
    output_dir="./output",
    max_seq_length=2048,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-5,
    logging_steps=10,
    save_steps=100,
    packing=True,
)

trainer = SFTTrainer(
    model="Qwen/Qwen3-0.6B",  # change this accordingly if the user asks you to
    args=config,
    dataset=dataset,
)
trainer.train()
