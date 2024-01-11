## Attention to Detail: Inter-Resolution Knowledge Distillation

The development of computer vision solutions for **gigapixel images** in digital pathology is hampered by significant
computational limitations due to the **large size** of whole slide images. An appealing solution is to use **smaller 
resolutions** during inference, and using **knowledge distillation** to enhance the model performance from large to 
reduced  image resolutions. We propose to distill this information by incorporating **attention maps** during training,
using *strictly positive gradients*, via the propsoed **AT+** term.

You can find more details in the following [manuscript](https://ieeexplore.ieee.org/abstract/document/10289941).

## Installation

* Install in your enviroment a compatible torch version with your GPU. For example:

```
conda create -n kd_env python=3.8 -y
conda activate kd_env
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```

* Clone and install requirements.

```
git clone https://github.com/cvblab/kd_resolution.git
cd kd_resolution
pip install -r requirements.txt
```

## Datasets

We propose using SICAPv2 for validating the proposed setting, in the challenging task of patch-level grading.

Please, download and prepare the following dataset as indicated:

1. Download the datasets.

* [SICAPv2](https://data.mendeley.com/datasets/9xxm58dvs3/2).

2. Set the dataset in the corresponding folder, at `./local_data/datasets/SICAPv2/`.

The proposed setting is largely general, and **we encourage you to apply it to any patch-level classification task**!

## Inter-Resolution Knowledge Distillation

1. Teacher model training at the largest image resolution available (i.e. 10x).
```
python main.py --epochs 20 --input_shape 512 --target_shape 512 --experiment 512_KD_0_FM_0_SR_0_AM_0
```
2. Student model training at smaller resolutions (e.g. 2.5x) - Please, note that the relative weight of FM and AT terms
should be empirically fixed through optimization in the validation set.
```
python main.py --epochs 20 --input_shape 512 --target_shape 128 --experiment 128_KD_0_FM_01_SR_0_AM_10 --experiment_name_teacher 512_KD_0_FM_0_SR_0_AM_0 --alpha_fm 0.1 --alpha_La 10.0
```

## Citation

If you find this repository useful, please consider citing this paper:
```
@article{InterResDistillation2023,
  title={Attention to Detail: Inter-Resolution Knowledge Distillation},
  author={Rocío del Amor and Julio Silva-Rodriguez and Adrián Colomer and Valery Naranjo},
  journal={European Signal Processing Conference (EUSIPCO)},
  year={2023}
}
```

## License

- **Code** is released under [MIT License](LICENSE)
