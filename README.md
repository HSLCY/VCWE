# VCWE

Codes and corpora for paper "VCWE: Visual Character-Enhanced Word Embeddings" (NAACL 2019)

## Requirement

* pytorch: 1.0.0
* python: 3.6.5
* numpy: 1.15.4

## Preparation

The input file is a plain corpus. The first line of the vocabulary file contains two numbers about the corpus, the first one is the number of lines and the second one is the number of tokens (repeatable). Each subsequent line contains a token and its frequencies.

For example:

```
# vocabulary.txt:
93788 187575421
鲁文 132
北朝 434
桑托斯省 120
赞美诗 129
皮利 150
应选 164
调味 675
人型 250
通通 293
鱼池 260
历险记 662
```

## Training

```
python trainer.py \
    --input_file ./data/zh_wiki.txt \
    --vocab_file ./data/vocabulary.txt \
    --line_batch_size 32 \
    --sample_batch_size 128 \
    --learning_rate 1e-3 \
    --num_train_epochs 50 \
    --output_dir ./embedding/ \
    --seed 12345
```


## Evaluation

Evaluate the results for word similarity.

```
python evaluation/all_wordsim.py embedding/zh_wiki_VCWE_ep50.txt evaluation/word-sim/
```

Results:
```
   Serial         Dataset            Num Pairs       Not found         Rho
     1           CH-297.txt             297              33          0.5582
     2         CH-RG-65.txt              65               0          0.7461
     3         CH-MC-30.txt              30               0          0.7765
     4           CH-240.txt             240              16          0.5554
```



