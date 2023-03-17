# 下载模型
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from pathlib import Path
BASE_DIR = str(Path().resolve().parent.parent)

# model name
# model_name = "distilgpt2"
model_name = "gpt2"
# model_name = "gpt2-medium"

# save path
model_path = os.path.join(BASE_DIR,"data/train_model/")
# load model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
# save model
tokenizer.save_pretrained(model_path + model_name)
model.save_pretrained(model_path + model_name)