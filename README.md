# KinGen
 
## Abstract

Kinases are critical regulators in numerous cellular processes, and their dysregulation is linked to various diseases, including cancer. Thus, protein kinases have emerged as major drug targets at present, with approximately a quarter to a third of global drug development efforts focusing on kinases. Additionally, deep learning based molecular generation methods have shown obvious advantages in improving the efficiency and success rate of drug discovery. However, many current molecular generation models face challenges in considering specific targets and generating molecules with desired properties, such as target related activity. Here, we developed a specialized and enhanced deep learning-based molecular generation framework named as KinGen, which is specially designed for the efficient generation of small molecule kinase inhibitors. By integrating reinforcement learning, transfer learning, and a specialized reward module, KinGen leverages a binding affinity prediction model as part of its reward function, which allows it to accurately guide the generation process towards biologically relevant molecules with high target activity. This approach not only ensures that the generated molecules have desirable binding properties but also improves the efficiency of molecular optimization. The results demonstrate that KinGen can generate structurally valid, unique, and diverse molecules. The generated molecules exhibit binding affinities similar to known inhibitors (with an average docking score of -9.5 kcal/mol), highlighting the model's strengths in generating compounds with enhanced activity and KinGen may act as a valuable tool for accelerating kinase-targeted drug discovery. 

## Installation

You can use the environment.yml and requirements.txt to create a new conda environment with all the necessary dependencies for KinGen:

```
git clone https://github.com/Shawn-Lau-lxm/KinGen
cd KinGen
conda env create -f environment.yml
conda activate kingen
pip install -f requirements.txt
```

## Usage

KinGen includes three sub-modules:

 1. train_prior.py: Used for pre-training on big dataset.

 2. train_agent.py: Used for generating new molecules with high affinity and novelty using the reinforcement learning algorithm.



Example of running the command:

```
python train_agent.py --target-protein-name EPIDERMAL_GROWTH_FACTOR_RECEPTOR --kinase-model-name 17_kinases --use-transfer --dataset davis
```
