# SentMAE

[SamuelYang/SentMAE](https://huggingface.co/SamuelYang/SentMAE): pre-trained on generic corpus(Wikipedia+BookCorpus)

[SamuelYang/SentMAE_BEIR](https://huggingface.co/SamuelYang/SentMAE_BEIR): pre-trained on generic corpus(Wikipedia+BookCorpus) and fine-tuned on MS MARCO

[SamuelYang/SentMAE_MSMARCO](https://huggingface.co/SamuelYang/SentMAE_MSMARCO): pre-trained on generic corpus(Wikipedia+BookCorpus), continuously pre-trained and fine-tuned on MS MARCO

## Installation


```
pip install beir

git clone https://github.com/staoxiao/RetroMAE.git
cd RetroMAE
pip install .
```

## Workflow

### Qucick test

```
python beir_test.py --dataset nfcorpus --split test --batch_size 128 \
                    --model_name_or_path SamuelYang/SentMAE_BEIR \
                    --pooling_strategy cls --score_function dot
```

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

### Finetune
```
python -m torch.distributed.launch --nproc_per_node 8 \
-m bi_encoder.run \
--output_dir {path to save ckpt} \
--model_name_or_path Shitao/RetroMAE \
--do_train  \
--corpus_file ./data/BertTokenizer_data/corpus \
--train_query_file ./data/BertTokenizer_data/train_query \
--train_qrels ./data/BertTokenizer_data/train_qrels.txt \
--neg_file ./data/train_negs.tsv 
```
For more details, please refer to [RetroMAE Bi-encoder](https://github.com/staoxiao/RetroMAE/blob/master/examples/msmarco/README.md).
