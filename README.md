# *Important Notification About Data Bug* #

* In the initial release, there was a bug in <code>data_2_terminology/train.tsv</code>. If you used this file, ***you must redownload the fixed file***, as the old file did not properly include all terminologies.

* I've added <code>diffcheck.py</code> so you can check that the only difference between the <code>data_2/train.tsv</code> and <code>data_2_terminology/train.tsv</code> is the 608 terminology pairs.

* The baseline results for using terminology has also been updated, but the differences from before are minor.

Sorry about the confusion!!

# Baseline for WMT21 Machine Translation using Terminologies Task

This is a baseline for the [WMT21 Machine Translation using Terminologies](http://www.statmt.org/wmt21/terminology-task.html) task. The task invites participants to explore methods to incorporate terminologies into either the training or the inference process, in order to improve both the accuracy and consistency of MT systems on a new domain. 

For the baseline, we consider the English-to-French translation task, and evaluation is performed on the [TICO-19 dataset](https://tico-19.github.io/), which is part of the overall evaluation for the task in WMT21.

## Model

The baseline finetunes OPUS-MT systems which are pre-trained on the OPUS parallel data using the Marian toolkit. We used the [Huggingface ported version](https://huggingface.co/Helsinki-NLP/opus-mt-en-fr) and used Huggingface + Pytorch to train.

## Datasets

In the challenge you are allowed to use any parallel or monolingual data from previous WMT shared tasks. However, to reduce training time & resources we finetuned the pre-trained English-to-French model from MarianMT (OPUS-MT) on the following datasets:

### Training Datasets

* [Medline](https://github.com/biomedical-translation-corpora/corpora)

 | Split  | Num. examples |
 | ----------- | ----------- |
 | Training1  | 614093 |
 | Training2  | 6540 |
  
* [Taus](https://md.taus.net/corona)

 | Split  | Num. examples |
 | ----------- | ----------- |
 | Train  | 885606 |
  
* [Terminologies](http://data.statmt.org/wmt21/terminology-task/)
 
 | Split  | Num. examples |
 | ----------- | ----------- |
 | Train  | 608 |

### Subsampled dataset for finetuning
For the intial training dataset, we have two subsampled versions:
* **data_2**

| Dataset  | Taus | Medline1 | Medline2 | Terminologies | Total |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| Num. examples  | 30000 | 30000 | 6540 | 0 | 66540 |

*Note that Medline2 has max. 6540 pairs after filtering empty examples*

* **data_2_terminology**

| Dataset  | Taus | Medline1 | Medline2 | Terminologies | Total |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| Num. examples  | 30000 | 30000 | 6540 | 608 | 67148 |

### Evaluation Dataset

During training, we evaluate on the dev set of TICO-19, and the final evaluation is performed on the test set of TICO-19. Note that the dev and test sets are on the smaller side.

| Split  | Num. examples |
| ----------- | ----------- |
| Dev  | 971 |
| Test  | 2100 |


## Baseline results

| Training Epochs  | Use terminology | Eval set | BLEU |
| ------------- | ------------- | ------------- | ------------- |
| 3 epochs  | No  | TICO-19 Dev  | 40.0991  |
| 3 epochs  | Yes  | TICO-19 Dev  | **40.3334**  |
| 3 epochs  | No  | TICO-19 Test  | 37.5342  |
| 3 epochs  | Yes  | TICO-19 Test  | **37.6491**  |
||||
| 10 epochs  | No  | TICO-19 Dev  | 39.9382  |
| 10 epochs  | Yes  | TICO-19 Dev  | **40.0829**  |
| 10 epochs  | No  | TICO-19 Test  | 37.4869  |
| 10 epochs  | Yes  | TICO-19 Test  | **37.579**  |

*Note that due to the small size of the data, these results can vary depending on various settings (hyperparameters, training epochs, etc.). However, generally the results should be better when using terminology than not.*

## Run the code on Colab 

You can run the baseline experiments from this [colab notebook](https://colab.research.google.com/drive/1udhccAi9VTYnl6ZcfkaDLX8YaXV4v4Q1?usp=sharing). To make further changes to the code, make sure to choose "Save a copy in Drive" to save an editable copy to your own Google Drive.

## TODO

- [x] code for generating the datasets will be added
- [ ] add editable transformer code

