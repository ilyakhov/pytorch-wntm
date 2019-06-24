- Train WNTM (using prepared data, sampled from wiki-150k):
```
python3 train.py -c=configs/config.json
```

- Train pLSA (using sklearn.datasets.fetch_20newsgroups data):
```
pythron3 train.py -c=configs/config_ndw.json
```
Examples of TopTokens for topics from __fetch_20newsgroups__ are [here](https://github.com/ilyakhov/pytorch-wntm/blob/master/n_dw/ndw_vocab1000_ntopics50_phi_reg-phi-0.1_reg-thet-1.0_epochs15_2stepsFalse_num_documents_passes1_epoch15_toptokens_0.txt).
Selected topics:

* topic_1 — god:0.105 | believe:0.029 | christians:0.027 | christian:0.024 | faith:0.023 | truth:0.023 | say:0.022 | moral:0.021 | religion:0.020 | morality:0.018 | objective:0.016 | atheists:0.016 | belief:0.016 | exist:0.015 | existence:0.015 | does:0.014 | human:0.013 | religious:0.013 | question:0.013 | atheism:0.012

* topic_7 — gov:0.038 | nasa:0.030 | research:0.026 | center:0.024 | problems:0.022 | april:0.020 | 20:0.019 | 30:0.018 | speed:0.015 | engine:0.014 | organization:0.014 | test:0.014 | 12:0.012 | subject:0.012 | 14:0.010 | old:0.010 | writes:0.009 | 32:0.009 | months:0.009 | help:0.009

* topic_8 — windows:0.049 | files:0.031 | image:0.029 | software:0.028 | edu:0.025 | file:0.023 | ftp:0.020 | data:0.019 | color:0.017 | program:0.016 | format:0.015 | os:0.014 | bit:0.014 | graphics:0.014 | images:0.013 | mac:0.012 | programs:0.012 | version:0.012 | available:0.012 | internet:0.012
