# Integrating Curricula with Replays
Authors: Ren Jie Tee, Mengmi Zhang

This repository contains the code to run experiments about integrating curricula with replays. Our [paper](https://ojs.aaai.org/index.php/AAAI-SS/article/view/27486) has been accepted in the AAAI 2023 Summer Symposium Series.

## Project Description
Humans engage in learning and reviewing processes with curricula when acquiring new skills or knowledge. This human learning behavior has inspired the integration of curricula with replay methods in continual learning agents. The goal is to emulate the human learning process, thereby improving knowledge retention and facilitating learning transfer. Existing replay methods in continual learning agents involve the random selection and ordering of data from previous tasks, which has shown to be effective. However, limited research has explored the integration of different curricula with replay methods to enhance continual learning.

Our study takes initial steps in examining the impact of integrating curricula  with replay methods on continual learning in three specific aspects: the interleaved frequency of replayed exemplars with training data, the sequence in which exemplars are replayed, and the strategy for selecting exemplars into the replay buffer. These aspects of curricula design align with cognitive psychology principles and leverage the benefits of interleaved practice during replays, easy-to-hard rehearsal, and exemplar selection strategy involving exemplars from a uniform distribution of difficulties.
Based on our results, these three curricula effectively mitigated catastrophic forgetting and enhanced positive knowledge transfer, demonstrating the potential of curricula in advancing continual learning methodologies.

## Setup

```
git clone https://github.com/ZhangLab-DeepNeuroCogLab/Integrating-Curricula-with-Replays

pip install -r requirements.txt
```

## Datasets
We used ciFAIR-10 and ciFAIR 100 from [cvJena/ciFAIR](https://cvjena.github.io/cifair/). Experiments for both datasets were slightly different.

In general, we varied 3 variables based on ideas in Curriculum Learning
- Interleave division
- Arrangement of replay samples
- Selection of replay samples

### ciFAIR-10
All ciFAIR-10 relevant things is stored in the ciFAIR-10 folder.

Within the Code folder is all the python files to run experiments. All experiments will save .pkl files to Results/\<Experiment Name\>/. The \<Experiment Name\>.ipynb files can be used to generate plots from the .pkl files.

Code/data is the ciFAIR-10 dataset, and Code/cifair.py is a pytorch interface for the dataset. All experiments could be ran with:
```
python <filename> <gpu> 
```
e.g.
```
python VaryReplayInstanceSeqPretrainedCifarez2hard.py 0
```


Here are the files for each experiments:
- To observe catastrophic forgetting (no replay methods used)
    - Code/CatastrophicForgetting.py
- To vary interleave divisions (randomly selection and shuffling the sequence of replay samples)
    - Code/VaryInterleave.py
    - Code/VaryInterleaveBaseline.py (Baseline means that everything is shuffled together randomly)
- To vary sequence of replay samples (keeping interleave division at 1). For Conf. Score, the [pretrained CIFAR-10 model](https://github.com/huyvnphan/PyTorch_CIFAR10) is stored in Code/cifar10_models.
    - Code/VaryReplayClassSeqClusterez2hard.py (By Class, Dist. Vec, Easy to Hard)
    - Code/VaryReplayClassSeqClusterhard2ez.py (By Class, Dist. Vec, Hard to Easy)
    - Code/VaryReplayClassSeqPretrainedCifarez2hard.py (By Class, Conf. Score, Easy to Hard)
    - Code/VaryReplayClassSeqPretrainedCifarhard2ez.py (By Class, Conf. Score, Hard to Easy)
    - Code/VaryReplayInstanceSeqClusterez2hard.py (By Instance, Dist. Vec, Easy to Hard)
    - Code/VaryReplayInstanceSeqClusterhard2ez.py (By Instance, Dist. Vec, Hard to Easy)
    - Code/VaryReplayInstanceSeqPretrainedCifarez2hard.py (By Instance, Conf. Score, Easy to Hard)
    - Code/VaryReplayInstanceSeqPretrainedCifarhard2ez.py (By Instance, Conf. Score, Hard to Easy)
    - Code/VaryReplaySeqBaseline.py (Randomly Sorting the Replay Samples)
- To vary the selection of replay samples (keeping interleave division at 1 and randomly shuffling the sequence of replay samples)
    - Code/VaryBufferSelectionClusterDistanceEasiest.py (Dist. Vec, Selecting Easiest Samples)
    - Code/VaryBufferSelectionClusterDistanceHardest.py (Dist. Vec, Selecting Hardest Samples)
    - Code/VaryBufferSelectionClusterDistanceUniform.py (Dist. Vec, Selecting Samples Uniformly Distributed Across Difficulty Levels)
    - Code/VaryBufferSelectionPretrainedCifarEasiest.py (Conf. Score, Selecting Easiest Samples)
    - Code/VaryBufferSelectionPretrainedCifarHardest.py (Conf. Score, Selecting Hardest Samples)
    - Code/VaryBufferSelectionPretrainedCifarUniform.py (Conf. Score, Selecting Samples Uniformly Distributed Across Difficulty Levels)

### ciFAIR-100
All ciFAIR-100 relevant things is stored in the ciFAIR-100 folder.

The Plots/\<Experiment Name\>.ipynb files can be used to generate plots from csv data downloaded from WandB.

Within the Code folder is all the python files to run experiments. All experiments will save relevant logs to WandB. To get started, follow step 1 and step 2 from [here](https://docs.wandb.ai/quickstart). Before running any experiments, run
```
python <filename> --help
```
e.g.
```
python CatastrophicForgetting.py --help
```
to see the relevant arguments that should be provided.

Code/data is the ciFAIR-100 dataset, and Code/cifair.py is a pytorch interface for the dataset. 

Here are the files for each experiments:
- Baselines
    - Code/CatastrophicForgetting.py (no replay methods used)
    - Code/Upperbound.py (Training with D<sub>1</sub> to D<sub>t</sub> at Task t)
    - Code/VanillaReplay.py (vanilla replay technique: selects replay samples randomly, shuffles replay samples and D<sub>t</sub> together randomly)
    - Code/FullDataset.py (training on the whole of ciFAIR-100 at one go)
- To vary interleave divisions (randomly selection and shuffling the sequence of replay samples)
    - Code/VaryDiv.py (Keeps track of total test accuracy of all seen classes)
    - Code/VaryDivFORGET.py (Keeps track of test accuracy of classes from only task 1)
    - For the few arguments you see from running `--help`, 
        - `seed` refers to the seed you set to randomise the controlled variables (how the replay samples are selected and their sequence)
        - `groupname` is used to organise your runs on WandB
        - `classseq` is a seed used to shuffle the sequence of classes for feedforward training in ciFAIR-100 (we used 100 for our experiments)
        - `buflen` refers to the number of samples stored in the buffer
        - `divnum` varies the division number
- To vary sequence of replay samples. For Conf. Score, we pretrained the exact same model on the whole of ciFAIR-100, using the FullDataset.py file. The resultant model is Code/pretrained
    - Code/VaryReplaySeq.py
    - Code/VaryReplaySeqFORGET.py
    - The naming convention (of including FORGET for the 2nd file) is the same as that for experiments varying interleave divisions.
    - The arguments are the same as that for experiments varying interleave divisions, except:
        - `seqpretrained` is used to determine whether you use Conf. Score (set this to 1 if you are using Conf. Score)
        - `seqbyclass` is used to determine whether you are sorting the samples by their class difficulty or instance difficulty (set this to 1 if you are sorting by class difficulty)
        - `seqhard2ez` is used to determine whether you are sorting from Easy to Hard or Hard to Easy (set this to 1 if you are sorting from Hard to Easy)
- To vary the selection of replay samples (randomly shuffling the sequence of replay samples)
    - Code/VaryBufferSel.py
    - This single file tracks both the total test accuracy of all seen classes and the test accuracy of classes from only task 1
    - The arguments are the same as that for the previous 2 experiments, except for:
        - `style`
            - `easiest`: Selecting the easiest samples
            - `hardest`: Selecting the hardest samples
            - `uniform`: Selecting Samples Uniformly Distributed Across Difficulty Levels
## Notes
The source code is for illustration purpose only. Specific reconfigurations may be needed to run certain scripts. We do not provide techinical supports but we would be happy to discuss about SCIENCE!
