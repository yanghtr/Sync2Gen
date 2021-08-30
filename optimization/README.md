# Generate prior
```
type=bedroom; # or living
data_path=../dataset_3dfront/data; # abs & rel dataset path
```

- `compute_num.py`: generate `num_unique.npy`
```
python compute_num.py --data_path $data_path --type $type
```

- `fit_pxv.py`: generate `Pxv.npy`
```
python fit_pxv.py --data_path $data_path --type $type
```

- `fit_pxe.py`: generate `Pxe.npy`
```
python fit_pxe.py --data_path $data_path --type $type
```

- `gen_trans_abs.py`: generate `trans_abs`, which will later be used to compute `Pcv_abs` parameters
```
python gen_trans_abs.py --data_path $data_path --type $type
```

- `gen_trans_rel.py`: generate `trans_rel`, which will later be used to compute `Pcv_rel` parameters
```
python gen_trans_rel.py --data_path $data_path --type $type
```

- `gen_pcv.py`: generate `Pcv_abs.npy` and `Pcv_rel.npy`
```
python gen_pcv.py --type $type
```

- `overlap.py`: generate `Povl.npy`
```
python overlap.py --data_path  $data_path --type $type
```

- `gen_prob_rotation_rel.py`: generate `Prot_rel.npy`
```
python gen_prob_rotation_rel.py --data_path  $data_path --type $type
```


