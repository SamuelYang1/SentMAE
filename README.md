# SentMAE

[SamuelYang/SentMAE](https://huggingface.co/SamuelYang/SentMAE): pre-trained on generic corpus(Wikipedia+BookCorpus)

[SamuelYang/SentMAE_BEIR](https://huggingface.co/SamuelYang/SentMAE_BEIR): pre-trained on generic corpus(Wikipedia+BookCorpus) and fine-tuned on MS MARCO

[SamuelYang/SentMAE_MSMARCO](https://huggingface.co/SamuelYang/SentMAE_MSMARCO): pre-trained on generic corpus(Wikipedia+BookCorpus), continuously pre-trained and fine-tuned on MS MARCO

## Installation


```
pip install beir

git clone https://github.com/staoxiao/RetroMAE
cd RetroMAE
pip install -e.

git https://github.com/texttron/tevatron
cd tevatron
pip install -e .

https://github.com/princeton-nlp/SimCSE
cd SimCSE
pip install -e .
```

## Workflow

### Pretrain
```
python -m torch.distributed.launch --nproc_per_node 8 \
  -m pretrain.run \
  --output_dir {path to save ckpt} \
  --data_dir {your data} \
  --do_train True \
  --model_name_or_path bert-base-uncased 
```
For more details, please refer to [RetroMAE Pre-training](https://github.com/staoxiao/RetroMAE/blob/master/examples/pretrain/README.md).

## For BEIR & MSMARCO
### Prepare data
```
cd tevatron/examples/coCondenser-marco
bash get_data.sh
```

### Finetune on MSMARCO
```
python -m tevatron.driver.train \  
  --output_dir ./SentMAE_BEIR \  
  --model_name_or_path SamuelYang/SentMAE \  
  --save_steps 20000 \  
  --train_dir ./marco/bert/train \
  --fp16 \  
  --per_device_train_batch_size 16 \  
  --learning_rate 2e-5 \  
  --num_train_epochs 4 \  
  --dataloader_num_workers 6
```
For more details, please refer to [tevatron](https://github.com/texttron/tevatron/tree/main/examples/coCondenser-marco).


### Test on BEIR

```
cd SentMAE
python beir_test.py --dataset nfcorpus --split test --batch_size 128 \
                    --model_name_or_path SamuelYang/SentMAE_BEIR \
                    --pooling_strategy cls --score_function dot
```

## For STS
### Prepare data
```
cd SimCSE/SentEval/data/downstream/
bash download_dataset.sh
```

### Finetune on NLI
```
python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train.py \
    --model_name_or_path SamuelYang/SentMAE \
    --train_file data/nli_for_simcse.csv \
    --output_dir SentMAE_NLI \
    --num_train_epochs 3 \
    --per_device_train_batch_size 128 \
    --learning_rate 5e-5 \
    --max_seq_length 64 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16
```

### Test on STS
```
python evaluation.py \
    --model_name_or_path SentMAE_NLI \
    --pooler cls \
    --task_set sts \
    --mode test
```
For more details, please refer to [SimCSE](https://github.com/princeton-nlp/SimCSE).
