# BackdoorDynamicDNN


This is the PyTorch implementation demo of the INFOCOM 2023 paper **Mind Your Heart: Stealthy Backdoor Attack on Dynamic Deep Neural Network in Edge Computing**.
In our paper, we propose a simple yet effective backdoor attack on the dynamic multi-exit DNN models deployed in edge computing via injecting stealthiness backdoors.

Our repository contains two parts:

- Training and evaluation code.
- Defense experiments used in the paper.




## Training code

Before training, the datasets (e.g., TinyImageNet and GTSRB) should be prepared in directory set in ```dataset_utils.py```.
Also, remember to specify result directory using variables in ```settings.py```.


The first step is to transform DNN to SDN, run command

```python
$ python train_sdn.py --dataset gtsrb --network resnet56 --pretrained_cnn model_cnn.pt --device cuda:0
```

The second step is to inject backdoor in DNN via SDN (with gtsrb and resnet56 as an example)

```python
$ python backdoor_sdn.py --dataset gtsrb --network resnet56 --pretrained model_sdn_25.pt --inj_rate 0.01 --backdoor_lr 0.0001 --backdoor_epoch 5 --device cuda:0 --copy copy1
```

After the second step, we can obtain a backdoored DNN transformed from SDN (which can be released on platforms and passes backdoor detection).

Finally, we simulate the victim's retraining SDN.
The backdoor should remains effective even after the retraining.

```python
$ python retrainic_sdn.py --dataset gtsrb --network resnet56 --pretrained_sdn copy1/model_backdoor_sdn_5.pt --retrain_ic_copy copy5 --device cuda:1
```

## Evaluation code

The example of  ASR and accuracy for different datasets and model structures are provided in `TestParamResNet.ipynb`.

## Defense experiments

We evaluated four defenses in our paper: 

- For **Neural Cleanse** Run the command

```python
python neural_cleanse.py --model resnet56 --dataset gtsrb --pretrained_sdn copy1/model_backdoor_sdn_5.pt --device cuda:0
```

The results will be printed and logged in `results/nc` folder.
Note that the computed Anomaly Index by NC may vary over different runs because of randomness.

- For **DF-TND**, we provide the example in `DFTND/dftnd.ipynb`
- For **STRIP**, we provide the example in `STRIP.ipynb`
- For **backdoor unlearning**, we provide the example in `I-BAU/cnn_detect.ipynb`
    
    After backdoor unlearning, run command to transform the DNN to SDN and activate the backdoor.
    
    ```python
    python retrain_BAU.py --dataset gtsrb --network resnet56 --pretrained_sdn BAU/BAU_cnn_10.pt --retrain_ic_copy epoch_10 --device cuda:0
    ```


## Requirements

The code has been tested with Python3.8, Pytorch1.10 and cuda11.1

## Citation

```
@inproceedings{backdoor_dynamic_dnn_2023,
  author    = {Tian Dong and
               Ziyuan Zhang and
               Han Qiu and
               Tianwei Zhang and
               Hewu Li and
               Terry Wang},
  title     = {Mind Your Heart: Stealthy Backdoor Attack on Dynamic Deep Neural Network in Edge Computing},
  booktitle = {{IEEE} {INFOCOM} 2023 - {IEEE} Conference on Computer Communications},
  publisher = {{IEEE}},
  year      = {2023}
}
```