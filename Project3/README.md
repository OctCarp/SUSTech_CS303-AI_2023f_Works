1. Prepare the Environment

`pip install -r requirements.txt`

2. Run evaluate_release.py

`python evaluate_release.py`

The output will look like this:
```
number of entities (containing items): 6729
number of relations: 7
100%|███████████████████████████████████████████████████████████████████████████████████████████| 30/30 [01:20<00:00,  2.68s/it]
(0.6359160858891864, 0.010230428932508886, False, False, False, False, 0.16842103004455566, 80.27808499336243, 0.10501384735107422, 20.869765281677246)
101.54402089118958
```

Each value in  (0.6359160858891864, 0.010230428932508886, False,        False,         False,       False,        0.16842103004455566, 80.27808499336243, 0.10501384735107422, 20.869765281677246)

corresponds to (auc,                ndcg5,                init_timeout, train_timeout, ctr_timeout, topk_timeout, init_time,           train_time,        ctr_time,          topk_time)