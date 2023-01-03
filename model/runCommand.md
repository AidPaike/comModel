```
python step2_trainModel.py \
    --model_name_or_path /root/Comfort_all/data/train_model/distilgpt2 \
    --train_file /root/Comfort_all/data/datasets/fzy_train1.txt \
    --per_device_train_batch_size 2 \
    --do_train \
    --output_dir /root/Comfort_all/data/train_model/distilgpt2_fzy1 \
    --save_steps 10000 \
    --save_total_limit 50 \
    --num_train_epochs 50 \
    --overwrite_output_dir True
```
```
python step2_trainModel.py \
    --model_name_or_path /root/Comfort_all/data/train_model/distilgpt2 \
    --train_file /root/Comfort_all/data/datasets/train_data_bos.txt \
    --per_device_train_batch_size 4 \
    --do_train \
    --output_dir /root/Comfort_all/data/train_model/distilgpt2_new \
    --save_steps 10000 \
    --save_total_limit 50 \
    --num_train_epochs 50 \
    --overwrite_output_dir True
```
```
python step2_trainModel.py \
    --model_name_or_path /root/Comfort_all/data/train_model/distilgpt2 \
    --train_file /root/Comfort_all/data/datasets/train_data_bos.txt \
    --validation_file /root/Comfort_all/data/datasets/val_data_bos.txt \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir /root/Comfort_all/data/train_model/distilgpt2_finetune \
    --save_steps 10000 \
    --save_total_limit 50 \
    --num_train_epochs 10 \
    --overwrite_output_dir True

```
```
python run_clm.py \
    --model_name_or_path ./model/distilgpt2 \
    --train_file ./datasets/train_data_bos.txt \
    --validation_file ./datasets/val_data_bos.txt \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir ./train_model/distilgpt2_finetune \
    --save_steps 20000 \
    --save_total_limit 50 \
    --num_train_epochs 100 \
    --overwrite_output_dir True
```

```
python run_clm.py \
    --model_name_or_path ./model/gpt2-medium \
    --train_file ./datasets/train_data_bos.txt \
    --validation_file ./datasets/val_data_bos.txt \
    --per_device_train_batch_siz 1 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --output_dir ./train_model/gpt2-medium_finetune \
    --save_steps 20000 \
    --save_total_limit 50 \
    --num_train_epochs 100
```



```
python run_generation.py \
    --model_name_or_path=./model/gpt2 \
    --prompt=hello
```



```
python run_generation.py \
    --model_name_or_path=./model/gpt2 \
    --prompt='function('

#生成内容
function( int x, string y ); const string s = new ArrayList< int, string >(); let cell = cell.type(); cell.start = 0; cell.end = 0; let char[] * = cell.argType;
```

```
python run_generation.py \
    --model_name_or_path=./train_model/distilgpt2_finetune/checkpoint-1000000 \
    --prompt='function(' \
    --num_return_sequences=2 \
    --length=512 \
    --k=0 \
    --p=0.9
```