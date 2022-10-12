### For Code-T5

prepare general codes
```bash
cd t5_backbone
cp ../BLEC ./
cp ../DataAugment ./
cp ../utils ./
cp ../multi-bleu.perl ./
```

prepare data
```bash
cp ../CD ./
cp ../logic2text ./
```

download pretrain model
```bash
cd codet5-base
sh download.sh
cd ../
```

train from scratch
```bash
python main.py --mode=train --edit-strategy=mix --fd
```

test on logic2text
```bash
python main.py --mode=test --load-ckpt=$load_ckpt
```
where `$load_ckpt` is the checkpoint choosed from the training log.

test on LCD
```bash
python main.py --mode=test --load-ckpt=$load_ckpt --data-path=CD/data
```

### For GPT-2

prepare general codes
```bash
cd gpt_backbone
cp ../BLEC ./
cp ../DataAugment ./
cp ../utils ./
cp ../multi-bleu.perl ./
```

prepare data
```bash
cp ../CD ./
cp ../logic2text ./
```

download pretrain model
```bash
cd gpt2
sh download.sh
cd ../
```

train from scratch
```bash
python main.py --mode=train --edit-strategy=mix --fd
```

test on logic2text
```bash
python main.py --mode=test --load-ckpt=$load_ckpt
```
where `$load_ckpt` is the checkpoint choosed from the training log.

test on LCD
```bash
python main.py --mode=test --load-ckpt=$load_ckpt --data-path=CD/data
```



