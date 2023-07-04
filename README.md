# Preprocess

Import the experimental virtual environment in conda
```
conda env -f mran.yaml
```
then enter the environment:
```
conda activate mran
```

### Process raw pathology images
Sample dataset download: [Link 1](https://drive.google.com/file/d/13Fb2U59KiXnhqLfjSwgZ3Vpr2uBf7c2I/view?usp=sharing) [Link 2](https://pan.baidu.com/s/1C-xuMsTrVKLYyEGONUocwg?pwd=pz75) 

After downloading, extract `example.zip` to the `MRAN/WSI/` directory, then preprocess the original image:
```
cd pre
python run_preprocess.py
```
The storage format of the original data set can refer to the sample data set in `MRAN/WSI/example ` : 
> MRAN/WSI/dataset_name/\*/slide-idx1.svs
> 
> ...
> 
> MRAN/WSI/dataset_name/\*/slide-idxn.svs

Because the example dataset comes from TCGA, the first 12 bits of the file name `slide_idxi.svs` of each image are its case id.


  
### Divide the dataset
```
cd pre
python pro_csv.py
```

The format of the label file of the original dataset can refer to `MRAN/csv/example/sheet/total.csv`:
| File Name | Sample Type |
|--|--|
| slide-idxi.svs | Primary Tumor |
| slide-idxj.svs |Solid Tissue Normal  |
|....|...|


# Train and test

### train

```
python main.py
```
`main.yaml` is used to set parameters.

### test
```
python test.py
```
The `predict.csv` of the directory `"output_dir"` records the prediction results on the test set.


# Interpretability experiment

```
cd interpretability
python top_bag_top_patch.py
```
`"output_dir"/tb_tp/*/predict.csv` records the prediction results on the test set.
