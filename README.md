# Investigating the Robustness of Natural Language Generation from Logical Forms via Counterfactual Samples

## Introduction
This repository contains the data and code for the paper **[Investigating the Robustness of Natural Language Generation from Logical Forms via Counterfactual Samples](https://arxiv.org/abs/2210.08548)**.
<br>Chengyuan Liu, Leilei Gan, Kun Kuang, Fei Wu</br>


## Requirements

* Python == 3.7
* `pip install -r requirements.txt`

We also rely on some external resources, you can manually download them and put them into corresponding directories.

- Download [pretrained Code-T5 checkpoints](https://huggingface.co/Salesforce/codet5-base), and put it into the ``t5_backbone/codet5-base`` directory. You can also run `t5_backbone/codet5-base/download.sh`.
- Download [pretrained GPT-2 checkpoints](https://huggingface.co/gpt2), and put it into the ``gpt_backbone/gpt2`` directory. You can also run `gpt_backbone/gpt2/download.sh`.

## Train Code-T5 Logic2Text Model

prepare general codes
```bash
cd t5_backbone
cp -r ../BLEC ./
cp -r ../DataAugment ./
cp -r ../utils ./
cp -r ../multi-bleu.perl ./
```

prepare data
```bash
cp -r ../CD ./
cp -r ../logic2text ./
```

train from scratch
```bash
python main.py --mode=train --edit-strategy=mix
```

test on logic2text
```bash
python main.py --mode=test --load-ckpt=$load_ckpt
```
where `$load_ckpt` is the checkpoint chosen from the training log.

test on LCD
```bash
python main.py --mode=test --load-ckpt=$load_ckpt --data-path=CD/data
```

### Train GPT-2 Logic2Text Model

prepare general codes
```bash
cd gpt_backbone
cp -r ../BLEC ./
cp -r ../DataAugment ./
cp -r ../utils ./
cp -r ../multi-bleu.perl ./
```

prepare data
```bash
cp -r ../CD ./
cp -r ../logic2text ./
```

train from scratch
```bash
python main.py --mode=train --edit-strategy=mix
```

test on logic2text
```bash
python main.py --mode=test --load-ckpt=$load_ckpt
```
where `$load_ckpt` is the checkpoint chosen from the training log.

test on LCD
```bash
python main.py --mode=test --load-ckpt=$load_ckpt --data-path=CD/data
```

## Contact
If you have any issues or questions about this repo, feel free to contact liucy1@zju.edu.cn.

## License
[Apache License 2.0](./LICENSE) 


## Citation

Please cite the following paper if you found our work useful. Thanks!

```bibtex
@inproceedings{liu-etal-2022-investigating,
    title = "Investigating the Robustness of Natural Language Generation from Logical Forms via Counterfactual Samples",
    author = "Liu, Chengyuan  and
      Gan, Leilei  and
      Kuang, Kun  and
      Wu, Fei",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.370",
    pages = "5499--5512",
}
```
