# IndicBART

_Pre-trained, multilingual sequence-to-sequence models for Indian languages_

You can read more about IndicBART [here](https://indicnlp.ai4bharat.org/indic-bart). IndicBART is part of the [AI4Bharat tools for Indian languages](https://indicnlp.ai4bharat.org).

## Installation

1. Install the [YANMTT toolkit](https://github.com/prajdabre/yanmtt). Make sure to create a new conda or virtual environment to ensure things work smoothly.
2. Download the following: 

    - **v1** [(Vocabulary)](https://storage.googleapis.com/ai4bharat-indicnlg-public/indic-bart-v1/albert-indicunified64k.zip) 
             [(Model)](https://storage.googleapis.com/ai4bharat-indicnlg-public/indic-bart-v1/indicbart_model.ckpt) 

3. Decompress the vocabulary zip: `unzip albert-indicunified64k.zip`

## Finetuning IndicBART for NMT

### Sample training corpora

- [en-bn.bn](https://storage.googleapis.com/ai4bharat-indicnlg-public/sample_data/train.en-bn.bn)
- [en-bn.en](https://storage.googleapis.com/ai4bharat-indicnlg-public/sample_data/train.en-bn.en)
- [en-hi.hi](https://storage.googleapis.com/ai4bharat-indicnlg-public/sample_data/train.en-hi.hi)
- [en-hi.hi](https://storage.googleapis.com/ai4bharat-indicnlg-public/sample_data/train.en-hi.en) 

### Sample development set

3-way parallel: [en](https://storage.googleapis.com/ai4bharat-indicnlg-public/sample_data/dev.en) [hi](https://storage.googleapis.com/ai4bharat-indicnlg-public/sample_data/dev.hi) [bn](https://storage.googleapis.com/ai4bharat-indicnlg-public/sample_data/dev.bn) 

### Script conversion

- The Indic side of the data needs to converted to the Devanagari script. This can be done using the [Indic NLP Library](https://github.com/anoopkunchukuttan/indic_nlp_library). _(Sample code coming soon)_
- The sample data provided above has already been converted to the Devanagari script, so you can use it as is. 

### Fine-tuning command

```
python PATH-TO-YANMTT/train_nmt.py --train_slang hi,bn --train_tlang en,en  \
    --dev_slang hi,bn --dev_tlang en,en --train_src train.en-hi.hi,train.en-bn.bn \
    --train_tgt train.en-hi.en,train.en-bn.en --dev_src dev.hi,dev.bn --dev_tgt dev.en,dev.en \
    --model_path model.ft --encoder_layers 6 --decoder_layers 6 --label_smoothing 0.1 \
    --dropout 0.1 --attention_dropout 0.1 --activation_dropout 0.1 --encoder_attention_heads 16 \
    --decoder_attention_heads 16 --encoder_ffn_dim 4096 --decoder_ffn_dim 4096 \
    --d_model 1024 --tokenizer_name_or_path albert-indicunified64k --warmup_steps 16000 \
    --weight_decay 0.00001 --lr 0.001 --max_gradient_clip_value 1.0 --dev_batch_size 128 \
    --port 22222 --shard_files --hard_truncate_length 256 --pretrained_model indicbart_model.ckpt &> log
```

At the end of training, you should find the model with the highest BLEU score for a given language pair. This will be model.ft.best_dev_bleu.<language>-en where language can be  hi or bn. The model training log will tell you the iteration number when the best performing checkpoint was last saved. <br>

### Decoding command
  
```
decmod=BEST-CHECKPOINT-NAME
    
python PATH-TO-YANMTT/decode_nmt.py --model_path $decmod --slang hi --tlang en \
    --test_src dev.hi --test_tgt dev.trans --port 23352 --encoder_layers 6 --decoder_layers 6 \
    --encoder_attention_heads 16 --decoder_attention_heads 16 --encoder_ffn_dim 4096 \
    --decoder_ffn_dim 4096 --d_model 1024 --tokenizer_name_or_path albert-indicunified64k \
    --beam_size 4 --length_penalty 0.8
```

### Notes:


1. If you want to use an IndicBART model with language specific scripts, we provide that variant as well: [(Vocabulary)](https://storage.googleapis.com/ai4bharat-indicnlg-public/indic-bart-v1/albert-indic64k.zip) [(Model)](https://storage.googleapis.com/ai4bharat-indicnlg-public/indic-bart-v1/separate_script_indicbart_model.ckpt) 
    
2. If you want to perform additional pre-training of IndicBART or train your own then follow the instructions in: https://github.com/prajdabre/yanmtt/blob/main/examples/train_mbart_model.sh 
    
3. For advanced training options, look at the examples in: https://github.com/prajdabre/yanmtt/blob/main/examples 
   
## Finetuning IndicBART for Summarization

### Sample Corpus

- [train document](https://storage.googleapis.com/ai4bharat-indicnlg-public/sample_data/train.text.hi)
- [train summary](https://storage.googleapis.com/ai4bharat-indicnlg-public/sample_data/train.summary.hi)
- [dev document](https://storage.googleapis.com/ai4bharat-indicnlg-public/sample_data/dev.text.hi)
- [dev summary](https://storage.googleapis.com/ai4bharat-indicnlg-public/sample_data/dev.summary.hi) 

### Fine-tuning command

```
python PATH-TO-YANMTT/train_nmt.py --train_slang hi --train_tlang hi --dev_slang hi --dev_tlang hi \
    --train_src train.text.hi --train_tgt train.summary.hi --dev_src dev.text.hi \
    --dev_tgt dev.summary.hi --model_path model.ft --encoder_layers 6 --decoder_layers 6 \
    --label_smoothing 0.1 --dropout 0.1 --attention_dropout 0.1 --activation_dropout 0.1 \
    --encoder_attention_heads 16 --decoder_attention_heads 16--encoder_ffn_dim 4096 \
    --decoder_ffn_dim 4096 --d_model 1024 --tokenizer_name_or_path albert-indicunified64k \
    --warmup_steps 16000 --weight_decay 0.00001 --lr 0.0003 --max_gradient_clip_value 1.0 \
    --dev_batch_size 128 --port 22222 --shard_files --hard_truncate_length 512 \
    --pretrained_model indicbart_model.ckpt --max_src_length 384 --max_tgt_length 40 \
    --is_summarization --dev_batch_size 64 --max_decode_length_multiplier -60 \
    --min_decode_length_multiplier -10 --no_repeat_ngram_size 4 --length_penalty 1.0 \
    --max_eval_batches 20 --hard_truncate_length 512 
```

### Decoding command
  
```
decmod=BEST-CHECKPOINT-NAME
    
python PATH-TO-YANMTT/decode_nmt.py --model_path $decmod --slang hi --tlang en \
    --test_src dev.text.hi --test_tgt dev.trans --port 23352 --encoder_layers 6 \
    --decoder_layers 6 --encoder_attention_heads 16 --decoder_attention_heads 16 \
    --encoder_ffn_dim 4096 --decoder_ffn_dim 4096 --d_model 1024 \
    --tokenizer_name_or_path albert-indicunified64k --beam_size 4 \
    --max_src_length 384 --max_decode_length_multiplier -60 --min_decode_length_multiplier -10 \
    --no_repeat_ngram_size 4 --length_penalty 1.0 --hard_truncate_length 512 
```  
    
## Contributors
    
- [Raj Dabre](mailto:prajdabre@gmail.com)
- Himani Shrotriya
- [Anoop Kunchukuttan](mailto:anoop.kunchukuttan@gmail.com)
- Ratish Puduppully 
- Mitesh M. Khapra  
- Pratyush Kumar

## Citing
    
If you use IndicBART in your work, please cite:

```
@misc{dabre2021indicbart,
      title={IndicBART: A Pre-trained Model for Natural Language Generation of Indic Languages}, 
      author={Raj Dabre and Himani Shrotriya and Anoop Kunchukuttan and Ratish Puduppully and Mitesh M. Khapra and Pratyush Kumar},
      year={2021},
      eprint={2109.02903},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
    }    
```   

## License
    
IndicBART is licensed under the MIT License    
