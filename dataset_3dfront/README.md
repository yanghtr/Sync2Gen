# 3D-FRONT Dataset

## 1. Download 3D-FRONT dataset from [https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset)
```
├──RAW_DATA_PATH
    ├── 3D-FRONT
    ├── 3D-FUTURE-model
    ├── categories.py
    ├── model_info.json
```
**Note:** 3D-FRONT is updated after the paper submission so the preprocessing is a little different from the original paper (marked as *DFP* below).


## 2. Extract the top 20 classes (Optional)
```
cd dataset_3dfront
```

- Generate `assets/cat2id_all.pkl`:
```
python gen_cat2id_all.py # assets/gen_cat2id_all.py
```
- Generate `assets/cat2id_bedroom.pkl` and `assets/cat2id_living.pkl`
```
type=bedroom; # or living
python json2layout_distribution.py --type $type
```

## 3. Generate the datasets for all room types
- Generate the split dataset:
```
type=bedroom; # or living
python json2layout.py --type $type --future_path RAW_DATA_PATH/3D-FUTURE-model --json_path RAW_DATA_PATH/3D-FRONT --model_info_path RAW_DATA_PATH/model_info.json
```

## 4. Filter and merge the datasets for each room type

For bedroom, we merge `Bedroom.npy`, `MasterBedroom.npy`, `SecondBedroom.npy` together. We filter out bedroom without bed. 4111 scenes in total. we use all for training (4000) and validation (111).
```
python mergeDataset_bedroom.py 
```

For livingroom, we merge `LivingRoom.npy`, `LivingDiningRoom.npy` together. 4684 scenes in total. We use all for training (4000) and validation (684, *DFP*). We filter out scenes with less than 6 objects (*DFP* 4 in the paper).
```
python mergeDataset_living.py 
```

`Bedroom_train_val.npy` and `Livingroom_train_val.npy` are generated in `./outputs` .

## 5. Generate the final abs & rel dataset
```
type=bedroom; # or living
python gen_3dfront_dataset.py --type $type
```

The final dataset is stored in `./data/bedroom` or `./data/living` .

Final dataset:
```
├──data
    ├── bedroom
       ├── 0_abs.npy
       ├── 0_rel.pkl
       ├── ...
    ├── living
       ├── 0_abs.npy
       ├── 0_rel.pkl
       ├── ...
    ├── train_bedroom.txt
    ├── train_living.txt
    ├── val_bedroom.txt
    └── val_living.txt
```



