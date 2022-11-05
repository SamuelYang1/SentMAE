# SentMAE

[SamuelYang/SentMAE](https://huggingface.co/SamuelYang/SentMAE): pre-trained on generic corpus(Wikipedia+BookCorpus)

[SamuelYang/SentMAE_BEIR](https://huggingface.co/SamuelYang/SentMAE_BEIR): pre-trained on generic corpus(Wikipedia+BookCorpus) and fine-tuned on MS MARCO

[SamuelYang/SentMAE_MSMARCO](https://huggingface.co/SamuelYang/SentMAE_MSMARCO): pre-trained on generic corpus(Wikipedia+BookCorpus), continuously pre-trained and fine-tuned on MS MARCO


**(New!!!) The Pretraining and finetuning code is merged to [RetroMAE](https://github.com/staoxiao/RetroMAE).**

### For testing:

```
pip install beir

python beir_test.py --dataset nfcorpus --split test --batch_size 128 \
                    --model_name_or_path SamuelYang/SentMAE_BEIR \
                    --pooling_strategy cls --score_function dot
```
