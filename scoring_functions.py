#!/usr/bin/env python
from __future__ import print_function, division
import numpy as np
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import AllChem
from rdkit import DataStructs
from sklearn import svm
import time
import pickle
import re
import threading
import pexpect
rdBase.DisableLog('rdApp.error')

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import os
import json
from src.utils import DrugTargetDataset, collate, AminoAcid, ci, r_squared_error
from src.models.DAT import DAT3

"""Scoring function should be a class where some tasks that are shared for every call
   can be reallocated to the __init__, and has a __call__ method which takes a single SMILES of
   argument and returns a float. A multiprocessing class will then spawn workers and divide the
   list of SMILES given between them.

   Passing *args and **kwargs through a subprocess call is slightly tricky because we need to know
   their types - everything will be a string once we have passed it. Therefor, we instead use class
   attributes which we can modify in place before any subprocess is created. Any **kwarg left over in
   the call to get_scoring_function will be checked against a list of (allowed) kwargs for the class
   and if a match is found the value of the item will be the new value for the class.

   If num_processes == 0, the scoring function will be run in the main process. Depending on how
   demanding the scoring function is and how well the OS handles the multiprocessing, this might
   be faster than multiprocessing in some cases."""

class no_sulphur():
    """Scores structures based on not containing sulphur."""

    kwargs = []

    def __init__(self):
        pass
    def __call__(self, smile):
        mol = Chem.MolFromSmiles(smile)
        if mol:
            has_sulphur = any(atom.GetAtomicNum() == 16 for atom in mol.GetAtoms())
            return float(not has_sulphur)
        return 0.0

class tanimoto():
    """Scores structures based on Tanimoto similarity to a query structure.
       Scores are only scaled up to k=(0,1), after which no more reward is given."""

    kwargs = ["k", "query_structure"]
    k = 0.7
    query_structure = "Cc1ccc(cc1)c2cc(nn2c3ccc(cc3)S(=O)(=O)N)C(F)(F)F"

    def __init__(self):
        query_mol = Chem.MolFromSmiles(self.query_structure)
        self.query_fp = AllChem.GetMorganFingerprint(query_mol, 2, useCounts=True, useFeatures=True)

    def __call__(self, smile):
        mol = Chem.MolFromSmiles(smile)
        if mol:
            fp = AllChem.GetMorganFingerprint(mol, 2, useCounts=True, useFeatures=True)
            score = DataStructs.TanimotoSimilarity(self.query_fp, fp)
            score = min(score, self.k) / self.k
            return float(score)
        return 0.0

class activity_model():
    """Scores based on an ECFP classifier for activity."""

    kwargs = ["clf_path"]
    clf_path = '/home/lydia/big_lydia_database/to_learn/REINVENT/data/clf.pkl'

    def __init__(self):
        with open(self.clf_path, "rb") as f:
            self.clf = pickle.load(f)

    def __call__(self, smile):
        mol = Chem.MolFromSmiles(smile)
        if mol:
            fp = activity_model.fingerprints_from_mol(mol)
            score = self.clf.predict_proba(fp)[:, 1]
            return float(score)
        return 0.0

    @classmethod
    def fingerprints_from_mol(cls, mol):
        fp = AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=True)
        size = 2048
        nfp = np.zeros((1, size), np.int32)
        for idx,v in fp.GetNonzeroElements().items():
            nidx = idx%size
            nfp[0, nidx] += int(v)
        return nfp
    
class fusion_dta_transfer():
    """Predict binding affinity between generated molecules and given target."""

    def __init__(self, predictor_saved_path, protein_sequence, dataset, kinase_model="ALK_TYROSINE_KINASE_RECEPTOR", transfer=True, cuda=True, embedding_dim=1280, rnn_dim=128, hidden_dim=256, graph_dim=256,
                 n_heads=8, dropout=0.3, alpha=0.2, pretrain=True, fold=10, scaffold=False, whole_set=True) -> None:

        self.model_path = predictor_saved_path
        self.protein_sequence = protein_sequence
        
        self.dataset = dataset
        self.use_cuda = cuda and torch.cuda.is_available()

        self.embedding_dim = embedding_dim
        self.rnn_dim = rnn_dim
        self.hidden_dim = hidden_dim
        self.graph_dim = graph_dim

        self.n_heads = n_heads
        self.dropout = dropout
        self.alpha = alpha

        self.is_pretrain = pretrain

        Alphabet = AminoAcid()

        self.kinase = kinase_model
        self.fold = fold
        self.is_scaffold = scaffold
        self.is_whole_set = whole_set
        self.use_transfer = transfer

    def __call__(self, smile:list):

        smile_num = len(smile)
        protein = [self.protein_sequence] * smile_num
        decoy_affinity = [torch.tensor(1)] * smile_num

        with open("predictor/data/kiba_pro_id.json", "r") as f:
            pid_dict = json.load(f)

        pid = [pid_dict[pro] for pro in protein]
        
        dataset_pred = DrugTargetDataset(smile, protein, decoy_affinity, pid, is_target_pretrain=self.is_pretrain, self_link=False,dataset=self.dataset)
        dataloader_pred = torch.utils.data.DataLoader(dataset_pred
                                                        , batch_size=smile_num
                                                        , shuffle=False
                                                        , collate_fn=collate
                                                        )
        #model
        model = DAT3(self.embedding_dim, self.rnn_dim, self.hidden_dim, self.graph_dim, self.dropout, self.alpha, self.n_heads, is_pretrain=self.is_pretrain)

        if self.use_transfer:
            model_list = []
            if self.is_whole_set:
                model_dir = os.path.join(self.model_path, self.kinase, self.dataset, "whole_set_cv")
            else:
                model_dir = os.path.join(self.model_path, self.kinase, self.dataset)
            for k in range(self.fold):
                if self.is_scaffold:
                    model_name = os.path.join(model_dir, "DAT_best_"+self.dataset+"_65smiles-scaffold-"+str(self.fold)+"fold"+str(k)+".pkl")
                    model.load_state_dict(torch.load(model_name)['model'], strict=False)
                    model_list.append(model)
                else:
                    model_name = os.path.join(model_dir, "DAT_best_"+self.dataset+"_65smiles-random-"+str(self.fold)+"fold"+str(k)+".pkl")
                    model.load_state_dict(torch.load(model_name)['model'], strict=False)
                    model_list.append(model)
        else:
            model_path = os.path.join(self.model_path, "DAT_best_"+self.dataset+"_65smiles.pkl")
            model.load_state_dict(torch.load(model_path)['model'], strict=False)

        if self.use_cuda:
            model.cuda()

        model.eval()
        with torch.no_grad():
            for pro, smi, aff in dataloader_pred:
                if self.use_cuda:
                    protein = [p.cuda() for p in pro]
                    smiles = [s.cuda() for s in smi]
                    affinity = aff.cuda()
                if self.use_transfer:
                    out_list = []
                    for k in range(self.fold):
                        model = model_list[k]
                        _, out = model(protein, smiles)
                        out_list.append(out)
                    ensemble_out_affinity = torch.stack(out_list, dim=1).mean(dim=1)
                    ensemble_out_affinity = ensemble_out_affinity.cpu()
                else:
                    _, pred_affinity = model(protein, smiles)
                    pred_affinity = pred_affinity.cpu()
        if self.use_transfer:
            return ensemble_out_affinity
        else:
            return pred_affinity

class Worker():
    """A worker class for the Multiprocessing functionality. Spawns a subprocess
       that is listening for input SMILES and inserts the score into the given
       index in the given list."""
    def __init__(self, scoring_function=None):
        """The score_re is a regular expression that extracts the score from the
           stdout of the subprocess. This means only scoring functions with range
           0.0-1.0 will work, for other ranges this re has to be modified."""

        self.proc = pexpect.spawn('./multiprocess.py ' + scoring_function,
                                  encoding='utf-8')

        print(self.is_alive())

    def __call__(self, smile, index, result_list):
        self.proc.sendline(smile)
        output = self.proc.expect([re.escape(smile) + " 1\.0+|[0]\.[0-9]+", 'None', pexpect.TIMEOUT])
        if output is 0:
            score = float(self.proc.after.lstrip(smile + " "))
        elif output in [1, 2]:
            score = 0.0
        result_list[index] = score

    def is_alive(self):
        return self.proc.isalive()

class Multiprocessing():
    """Class for handling multiprocessing of scoring functions. OEtoolkits cant be used with
       native multiprocessing (cant be pickled), so instead we spawn threads that create
       subprocesses."""
    def __init__(self, num_processes=None, scoring_function=None):
        self.n = num_processes
        self.workers = [Worker(scoring_function=scoring_function) for _ in range(num_processes)]

    def alive_workers(self):
        return [i for i, worker in enumerate(self.workers) if worker.is_alive()]

    def __call__(self, smiles):
        scores = [0 for _ in range(len(smiles))]
        smiles_copy = [smile for smile in smiles]
        while smiles_copy:
            alive_procs = self.alive_workers()
            if not alive_procs:
               raise RuntimeError("All subprocesses are dead, exiting.")
            # As long as we still have SMILES to score
            used_threads = []
            # Threads name corresponds to the index of the worker, so here
            # we are actually checking which workers are busy
            for t in threading.enumerate():
                # Workers have numbers as names, while the main thread cant
                # be converted to an integer
                try:
                    n = int(t.name)
                    used_threads.append(n)
                except ValueError:
                    continue
            free_threads = [i for i in alive_procs if i not in used_threads]
            for n in free_threads:
                if smiles_copy:
                    # Send SMILES and what index in the result list the score should be inserted at
                    smile = smiles_copy.pop()
                    idx = len(smiles_copy)
                    t = threading.Thread(target=self.workers[n], name=str(n), args=(smile, idx, scores))
                    t.start()
            time.sleep(0.01)
        for t in threading.enumerate():
            try:
                n = int(t.name)
                t.join()
            except ValueError:
                continue
        return np.array(scores, dtype=np.float32)

class Singleprocessing():
    """Adds an option to not spawn new processes for the scoring functions, but rather
       run them in the main process."""
    def __init__(self, scoring_function=None):
        self.scoring_function = scoring_function()
    def __call__(self, smiles):
        scores = [self.scoring_function(smile) for smile in smiles]
        return np.array(scores, dtype=np.float32)

def get_scoring_function(scoring_function, num_processes=None, **kwargs):
    """Function that initializes and returns a scoring function by name"""
    scoring_function_classes = [no_sulphur, tanimoto, activity_model, fusion_dta_transfer]
    scoring_functions = [f.__name__ for f in scoring_function_classes]
    scoring_function_class = [f for f in scoring_function_classes if f.__name__ == scoring_function][0]

    if scoring_function not in scoring_functions:
        raise ValueError("Scoring function must be one of {}".format([f for f in scoring_functions]))

    for k, v in kwargs.items():
        if k in scoring_function_class.kwargs:
            setattr(scoring_function_class, k, v) # set attribute v to k of scoring_function_class

    if num_processes == 0:
        return Singleprocessing(scoring_function=scoring_function_class)
    return Multiprocessing(scoring_function=scoring_function, num_processes=num_processes)
