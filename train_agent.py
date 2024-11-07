#!/usr/bin/env python

import torch
import math
import numpy as np
import time
import os
import argparse
from shutil import copyfile
from rdkit import Chem
from rdkit.Chem import QED

from model import RNN
from data_structs import Vocabulary, Inception
from scoring_functions import fusion_dta_transfer
from utils import Variable, seq_to_smiles, fraction_valid_smiles, unique, get_protein_sequence

def train_agent(restore_prior_from='data/Prior_chembl_31.ckpt',
                restore_agent_from='data/Prior_chembl_31.ckpt',
                dataset="davis",
                target_protein_name: str = None,
                target_protein_model_path: str = "predictor/saved_models",
                kinase_model_name: str = "ALK_TYROSINE_KINASE_RECEPTOR",
                use_transfer: bool = True,
                save_dir="data/results", voc_file = "data/Voc-chembl_31",
                learning_rate=0.0005,
                batch_size=64, n_steps=3000,
                num_processes=0, sigma=60,
                experience_replay=0, early_stop=20):

    voc = Vocabulary(init_from_file=voc_file)

    start_time = time.time()

    Prior = RNN(voc)
    Agent = RNN(voc)

    logger_time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    if use_transfer:
        save_dir = os.path.join(save_dir, target_protein_name, dataset, kinase_model_name,
                                "run_"+logger_time+"_Sigma"+str(sigma)+"_EarlyStop"+str(early_stop)
                                +"_ExReplay"+str(experience_replay)+"_BatchSize"+str(batch_size)+"_LearningRate"+str(learning_rate)+"_nSteps"+str(n_steps))
    else:
        save_dir = os.path.join(save_dir, target_protein_name, dataset, "fusion_dta",
                                "run_"+logger_time+"_Sigma"+str(sigma)+"_EarlyStop"+str(early_stop)
                                +"_ExReplay"+str(experience_replay)+"_BatchSize"+str(batch_size)+"_LearningRate"+str(learning_rate)+"_nSteps"+str(n_steps))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    copyfile('train_agent.py', os.path.join(save_dir, "train_agent.py"))

    # By default restore Agent to same model as Prior, but can restore from already trained Agent too.
    # Saved models are partially on the GPU, but if we dont have cuda enabled we can remap these
    # to the CPU.
    if torch.cuda.is_available():
        Prior.rnn.load_state_dict(torch.load(restore_prior_from))
        Agent.rnn.load_state_dict(torch.load(restore_agent_from))
    else:
        Prior.rnn.load_state_dict(torch.load(restore_prior_from, map_location=lambda storage, loc: storage))
        Agent.rnn.load_state_dict(torch.load(restore_agent_from, map_location=lambda storage, loc: storage))

    # We dont need gradients with respect to Prior
    for param in Prior.rnn.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(Agent.rnn.parameters(), lr=learning_rate)

    # Scoring_function
    target_protein_sequence = get_protein_sequence(target_protein_name)
    scoring_function = fusion_dta_transfer(target_protein_model_path, target_protein_sequence, dataset, kinase_model_name, use_transfer)

    # For policy based RL, we normally train on-policy and correct for the fact that more likely actions
    # occur more often (which means the agent can get biased towards them). Using experience replay is
    # therefor not as theoretically sound as it is for value based RL, but it seems to work well.
    experience = Inception(voc)

    # Information for the logger
    step_score = [[], []]

    print("Model initialized, starting training...")

    best_loss, early_stop_count = math.inf, 0
    for step in range(n_steps):

        # Sample from Agent
        seqs, agent_likelihood, entropy = Agent.sample(batch_size)

        # Remove duplicates, ie only consider unique seqs
        unique_idxs = unique(seqs)
        seqs = seqs[unique_idxs]
        agent_likelihood = agent_likelihood[unique_idxs]
        entropy = entropy[unique_idxs]

        # Get prior likelihood and score
        prior_likelihood, _ = Prior.likelihood(Variable(seqs))
        smiles = seq_to_smiles(seqs, voc) # len(smiles) = batch_szie
        print("Calculating generated smiles score~")
        #smiles = [smi for smi in smiles if smi != "" and Chem.MolFromSmiles(smi) is not None]

        score_affinity = scoring_function(smiles) # score.shape = (batch_size,)
        score = [max(_score, 5) / 5 for _score in score_affinity] # diversity filter
        print("The length of score--1: ", len(score))
        print("Calculating QED score~")
        #qed_score = [QED.qed(Chem.MolFromSmiles(smi)) if Chem.MolFromSmiles(smi) is not None else 0 for smi in smiles]
        qed_score = []
        for smi in smiles:
            if smi == "":
                qed_score.append(0)
            else:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    try:
                        qed_score.append(QED.qed(mol))
                    except:
                        qed_score.append(0)
                        with open(save_dir+"qed_error.log", "a") as f:
                            f.write(smi+"\n")
                else:
                    qed_score.append(0)
        print("The length of qed_score: ", len(qed_score))
        score = [_affinity if _qed >= 0.34 else 0 for _affinity, _qed in zip(score, qed_score)]
        print("The length of score: ", len(score))


        score, scaffold, scaf_fp = experience.update_score(smiles, score)

        # Calculate augmented likelihood
        augmented_likelihood = prior_likelihood + sigma * Variable(score)
        loss = torch.pow((augmented_likelihood - agent_likelihood), 2) # loss.shape = torch.size([batch_size])

        # Experience Replay
        # First sample
        if experience_replay and len(experience)>4:
            exp_seqs, exp_score, exp_prior_likelihood = experience.sample(4)
            exp_agent_likelihood, exp_entropy = Agent.likelihood(exp_seqs.long())
            exp_augmented_likelihood = exp_prior_likelihood + sigma * exp_score
            exp_loss = torch.pow((Variable(exp_augmented_likelihood) - exp_agent_likelihood), 2)
            loss = torch.cat((loss, exp_loss), 0)
            agent_likelihood = torch.cat((agent_likelihood, exp_agent_likelihood), 0)

        experience.add_experience(smiles, score, prior_likelihood, scaffold, scaf_fp)

        # Calculate loss
        loss = loss.mean() # torch.Size([])

        # Calculate gradients and make an update to the network weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mean_train_loss = loss.detach().item()

        # Convert to numpy arrays so that we can print them
        augmented_likelihood = augmented_likelihood.data.cpu().numpy()
        agent_likelihood = agent_likelihood.data.cpu().numpy()

        # Print some information for this step
        time_elapsed = (time.time() - start_time) / 3600
        time_left = (time_elapsed * ((n_steps - step) / (step + 1)))
        print("\n       Step {}   Fraction valid SMILES: {:4.1f}  Time elapsed: {:.2f}h Time left: {:.2f}h".format(
              step, fraction_valid_smiles(smiles) * 100, time_elapsed, time_left))
        print("  Agent    Prior   Target   Score             SMILES")
        for i in range(10):
            print(" {:6.2f}   {:6.2f}  {:6.2f}  {:6.2f}     {}".format(agent_likelihood[i],
                                                                       prior_likelihood[i],
                                                                       augmented_likelihood[i],
                                                                       score[i],
                                                                       smiles[i]))
        # Need this for Vizard plotting
        step_score[0].append(step + 1)
        step_score[1].append(np.mean(score))
        print(f'Step [{step+1}/{n_steps}]: Train loss: {mean_train_loss:.4f}')

        # Save the model
        if mean_train_loss < best_loss:
            best_loss = mean_train_loss
            torch.save(Prior.rnn.state_dict(), os.path.join(save_dir, 'Agent.ckpt'))
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1
        if early_stop_count >= early_stop:
            print('\nModel is not improving, so we halt the training session.')
            break

    # Save files
    experience.save_memory(os.path.join(save_dir, "memory.csv"))
    with open(os.path.join(save_dir, 'step_score.csv'), 'w') as f:
        f.write("step,score\n")
        for s1, s2 in zip(step_score[0], step_score[1]):
            f.write(str(s1) + ',' + str(s2) + "\n")

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--target-protein-name', default='ALK_TYROSINE_KINASE_RECEPTOR', help='Give target protein name.')
    parser.add_argument('--kinase-model-name', default='ALK_TYROSINE_KINASE_RECEPTOR', help='Give fusion dta transferring kinase model name.')
    parser.add_argument('--use-transfer', action='store_true', help='use fusion dta transerring model or not.')
    parser.add_argument('--dataset', default='davis', help='dataset: davis or kiba.')
    parser.add_argument('--cuda-device', default='0', help='Select cuda device ID.')
    parser.add_argument('--restore-prior-path', default='data/Prior_chembl_31.ckpt', help='Path storing prior model.')
    parser.add_argument('--restore-agent-path', default='data/Prior_chembl_31.ckpt', help='Path storing agent model.')
    parser.add_argument('--save-dir', default='data/results/', help='Save directory')
    parser.add_argument('--voc-file', default='data/Voc-chembl_31', help='Voc file path.')
    parser.add_argument('--sigma', default=7, type=int, help='Set sigma value.')
    parser.add_argument('--early-stop', default=400, type=int, help='Early stop steps.')
    parser.add_argument('--n-steps', default=3000, type=int, help='Step numbers.')
    parser.add_argument('--learning-rate', default=0.0005, type=float, help='Learning rate.')
    parser.add_argument('--batch-size', default=128, type=int, help='Batch size.')
    parser.add_argument('--experience-replay', default=1, type=int, help='Replay experience numbers.')
    parser.add_argument('--num-processes', default=0, type=int, help='Process numbers.')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    s = time.time()
    train_agent(restore_prior_from=args.restore_prior_path,
                restore_agent_from=args.restore_agent_path,
                dataset=args.dataset,
                target_protein_name=args.target_protein_name,
                kinase_model_name=args.kinase_model_name,
                use_transfer=args.use_transfer,
                save_dir=args.save_dir, voc_file=args.voc_file,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size, n_steps=args.n_steps,
                num_processes=args.num_processes,
                sigma=int(args.sigma), experience_replay=args.experience_replay, early_stop=args.early_stop)
    e = time.time()
    print("Use time: {:.4f}s".format(e - s))
