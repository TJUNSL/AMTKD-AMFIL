# Adaptive Multi-Teacher Knowledge Distillation based Android Malware Family Incremental Learning Framework

With the increasing number of Android malware, effectively classifying and identifying them has become an essential issue in the field of security. Traditional machine learning methods often require retraining models when faced with newly emerging malware families, which is time-consuming and inefficient. Methods based on replay samples or fine-tuning can significantly reduce the detection performance of models. Therefore, there is an urgent need to explore more efficient classification methods to address this challenge. This paper proposes an incremental learning framework for Android malware family classification based on adaptive multi-teacher knowledge distillation. This framework achieves effective incremental classification of Android malware families through dynamically expandable model representations. While ensuring good classification capabilities for known malware families, it can utilize a small number of samples from new malware families to enable the model to classify newly emerging families. Specifically, in each incremental task, the previously learned representations are frozen, additional feature dimensions are added from new learnable feature extractors, and knowledge distillation is performed by extracting from the initial and subsequent teacher models. However, drawing knowledge from two teacher models may lead to redundant predictions in the student model. To address this issue, we introduce pruning methods to eliminate redundant parameters in the feature extractor, effectively mitigating the potential parameter explosion problem caused by dynamic model expansion. Extensive results on four datasets demonstrate that the proposed class incremental framework outperforms strong competitors, showing significant advantages in dealing with newly emerging Android malware families.

## How To Use

### Run experiment

1. Edit the `[MODEL NAME].json` file for global settings.
2. Edit the hyperparameters in the corresponding  `[MODEL NAME].py` file (e.g., models/icarl.py).
3.  Run： `python main.py --config=./exps/[MODEL NAME].json`
   To run our method  `python main.py --config=./exps/mtkdcilf.json`

### Datasets

There are four Android malware datasets, two of which are widely used Android malware family datasets (**Drebin, AMD**) the other two are based on **VirusShare** datasets (**VirusShareImg** **VirusShareYearsImg**).

**The dataset can be accessed via [here]().**

Before proceeding with the experiment, set up how the dataset is loaded in `utils/data_manager.py`:

In the **_setup_data** method, see the comment.

## Acknowledgments

This work is based on [PyCIL](https://github.com/G-U-N/PyCIL).

## Contact

If you have any questions, please contact the author **Hongpeng Bai** (bai931214@tju.edu.cn).