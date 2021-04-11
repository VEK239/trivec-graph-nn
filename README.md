[Original JB Internship Task Repository](https://github.com/jbr-ai-labs/trivec_internship)
# Data split
New data split is located in notebooks/new_data_split.ipynb. The new split has no triples where from vertex or to vertex index is more than 630.
All those triples are shuffled and splitted between valid/test datasets.

The new data split is located in new_data/*.

There are also files polyphar_test_seen.csv and  polyphar_test_new.csv. Those files contain the test data triples which 
were seen by the model while training and weren't respectively. 
# Training and evaluating
Training/evaluating the TriVec model is located in notebooks/train_test_trivec.ipynb.
The trivec_test.py which is used in the notebook is made to test the model using the triples with seen and unseen pharmacies.
It requires a trained model checkpoint (--model_path param).


# Train logging
We use [neptune](https://neptune.ai/) to track metrics and losses during training.

# Usage
To train model only on drug-drug data (without proteins) use command
```
$ python run_trivec.py --metrics_separately --gpu --epoch 25
```
* ```--use_proteins``` for train with proteins.
* ```--log``` and ```--neptune_project [PROJECT_NAME]``` to tracking metrics and loses with neptune.
* ```--experiment_name '<EXP_NAME>'``` to add custom experiment name for neptune and model saving path.
To test model only on drug-drug data (without proteins) use command
```
$ python test_trivec.py --metrics_separately --model_path <PATH-TO-CHECKPOINT>
```