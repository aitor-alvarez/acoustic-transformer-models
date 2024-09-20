## Acoustic Transformer Models

The code in this repository can be used to finetune Transformer models such as Wav2vec or HuBert in downstream classification tasks. A classification head has been added to the models to be able to use it in acoustic classification tasks.

The following pretrained models (and other from the same family: https://huggingface.co/collections/facebook/xlsr-651e8a5bb947065cccb62c6c or https://huggingface.co/collections/facebook/wav2vec-20-651e865258e3dee2586c89f5) can be used and passed to the trainer:

```
facebook/wav2vec2-base
facebook/wav2vec2-large-xlsr-53
facebook/wav2vec2-xls-r-300m
facebook/wav2vec2-xls-r-1b
facebook/wav2vec2-xls-r-2b
facebook/hubert-large-ll60k
facebook/hubert-xlarge-ls960-ft
```

You can also pretrain a model directly by passing your own (--model_name).

The following command trains and tests the model:

```
python main.py --model_name 'either_pretrain_from_the_above_list_or_your_own_model'
--batch_size 32 --num_epochs 100
--data_dir 'path_to_dataset_directory'
--lang 'es' --n_gpus 4 --n_nodes 1 --strategy="ddp"
```
