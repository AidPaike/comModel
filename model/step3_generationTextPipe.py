#!/usr/bin/env python
# coding=utf-8

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import time
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def generationTextPipe(model_name_or_path="/root/Comfort_all/data/train_model/distilgpt2_finetune/checkpoint-160000",
                       prefixList=["""function("""],
                       num_return_sequences=50,
                       max_length=512,
                       temperature=1,
                       p=0.9,
                       k=0,
                       ):
    start = time.time()

    print(f'正在加载模型,大约需要10秒,请稍等')

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer, device=0)
    print("模型加载完成：", time.time() - start)

    allGeneration = generator(prefixList, num_return_sequences=num_return_sequences, max_length=max_length,
                              pad_token_id=tokenizer.eos_token_id, temperature=temperature, k=k, p=p)
    allFunctions = []

    for generationItem in allGeneration:
        for idx, item in enumerate(generationItem):
            # print('-'*30+'NO.'+ str(idx+1)+'-'*30)
            # print(item['generated_text'])
            allFunctions.append(item['generated_text'])
    return allFunctions


if __name__ == '__main__':
    prefixList = []

    prefix1 = """function("""
    prefix2 = """function(a,"""
    # prefix3 = """function(b,"""
    # prefix4 = """function(c,"""
    # prefix5 = """function(d,"""
    # prefix6 = """function(e,"""
    # prefix7 = """function(f,"""
    # prefix8 = """function(g,"""
    # prefix9 = """function(h,"""
    # prefix10 = """function(i,"""
    # prefix11 = """function(g,"""
    # prefix12 = """function(h,"""
    # prefix13 = """function(q,"""
    # prefix14 = """function(w,"""
    # prefix15 = """function(e,"""
    prefixList.append(prefix1)
    prefixList.append(prefix2)
    # prefixList.append(prefix4)
    # prefixList.append(prefix5)
    # prefixList.append(prefix6)
    # prefixList.append(prefix7)
    # prefixList.append(prefix8)
    # prefixList.append(prefix9)
    # prefixList.append(prefix10)
    # prefixList.append(prefix11)
    # prefixList.append(prefix12)
    # prefixList.append(prefix13)
    # prefixList.append(prefix14)
    # prefixList.append(prefix15)
    generationTextPipe(prefixList=prefixList)
