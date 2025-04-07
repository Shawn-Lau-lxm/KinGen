import pandas as pd
import numpy as np
import random
import sys
import os
import glob
import shutil
import argparse
from typing import List
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Scaffolds import MurckoScaffold


def read_kinase(filename: str, filetype: str="tsv", keep_affinity="IC50", outfile: str=None):
    """
    Read kinase data from BindingDB, and add column "comparing_symbol" to depict range value with "pIC50".
    """
    target_name = filename.split(".")[0]

    if filetype == "tsv":
        data = pd.read_csv(filename, sep="\t")
    elif filetype == "csv":
        data = pd.read_csv(filename)
    if outfile is None:
        if not os.path.exists(target_name):
            os.mkdir(target_name)
        outfile = os.path.join(target_name, "preprocess_" + filename)

    data["Organism"] = data["Target Source Organism According to Curator or DataSource"]
    organism_idx = data[(data.Organism != "Homo sapiens")].index.tolist()
    data = data.drop(index=organism_idx)
    
    data = data.dropna(axis=0,subset = ["Ligand SMILES"])

    if keep_affinity == "IC50":
        affinity_col = "IC50 (nM)"

    data = data.dropna(axis=0,subset = [affinity_col])
    data = data.reset_index(drop=True)

    data.insert(loc=10, column="comparing_symbol", value=0)
    for i in range(len(data)):
        if data.loc[i, affinity_col].split()[0].startswith(">"):
            data.loc[i, "comparing_symbol"] = 1
            data.loc[i, affinity_col] = data.loc[i, affinity_col].split(">")[-1]
        elif data.loc[i, affinity_col].split()[0].startswith("<"):
            data.loc[i, "comparing_symbol"] = 2
            data.loc[i, affinity_col] = data.loc[i, affinity_col].split("<")[-1]

    data = data[data[affinity_col].notnull()].copy()
    data[affinity_col] = pd.to_numeric(data[affinity_col], errors="coerce")
    data.insert(loc=11, column="pIC50", value=-np.log10((data[affinity_col]+1e-7)*1e-9))
    
    data.to_csv(outfile, sep="\t", index=False)
    return data, outfile


def keep_duplicate_mean(filename: str=None, outfile: str=None):
    """
    Retain the mean value of duplicate entries and generate tsv tables.
    """
    target_name = filename.split(".")[0]

    seq = kinase_dict(protein_name=target_name, dtype="SEQ")
    protein_id = kinase_dict(protein_name=target_name, dtype="KIBA_ID")

    fusion_chars = '#%)(+-.1032547698=ACBEDGFIHKMLONPSRUTWVY[Z]_acbedgfihmlonsruty@/\\'
    fusion_chars_set = set(fusion_chars)

    if outfile is None:
        if not os.path.exists(target_name):
            os.mkdir(target_name)
        outfile = os.path.join(target_name, "keepmean-norepeat_" + filename)
    
    data, _ = read_kinase(filename=filename)

    idx = data[(data.comparing_symbol != 0)].index.tolist()
    data = data.drop(index=idx)
    data = data.reset_index(drop=True)

    entries = []
    redundant = []
    dunk = []
    for i in range(len(data)):
        smiles = data["Ligand SMILES"][i]
        smiles_set = set(smiles)
        if len(smiles_set - fusion_chars_set) > 0:
            continue
        protein_number = data["Number of Protein Chains in Target (>1 implies a multichain complex)"][i]
        pro = data["BindingDB Target Chain  Sequence"][i].upper()
        protein = None
        if pro == seq:
            protein = pro
        else:
            for j in range(protein_number-1):
                pro = data["BindingDB Target Chain  Sequence."+str(j+1)][i].upper()
                if pro == seq:
                    protein = pro
        if not protein:
            continue
        affinity = data["pIC50"][i]
        ligand = data["BindingDB Reactant_set_id"][i]  # Add ligand id   
        if [smiles, protein, affinity] not in dunk:
            dunk.append([smiles, protein, affinity])
            entries.append([smiles, ligand, protein, affinity, protein_id])
        else:
            redundant.append([smiles, ligand, protein, affinity, protein_id])

    fusion_test = pd.DataFrame(entries, columns=["compound_iso_smiles", "BindingDB_reactant_id", "target_sequence", "affinity", "protein_id"])

    f3 = fusion_test.astype({"BindingDB_reactant_id":"str", "protein_id":"str"})
    d_rows = f3[f3.duplicated(subset=["compound_iso_smiles","target_sequence"],keep=False)]
    f4 = f3.drop(d_rows.index, axis=0)

    d_rows["dt"] = d_rows["compound_iso_smiles"].str.cat([d_rows["target_sequence"]], sep=" ")
    f5 = d_rows.groupby("dt").mean(numeric_only=True) 
    f5["dt"] = f5.index
    f5.index = range(len(f5))
    f5["compound_iso_smiles"] = f5["dt"].str.split().str[0]
    f5["target_sequence"] = f5["dt"].str.split().str[1]

    d_rows_dict_lig = d_rows.set_index("compound_iso_smiles")["BindingDB_reactant_id"].to_dict()
    f5["BindingDB_reactant_id"] = f5["compound_iso_smiles"].apply(lambda x: d_rows_dict_lig[x])
    d_rows_dict_pro = d_rows.set_index("target_sequence")["protein_id"].to_dict()
    f5["protein_id"] = f5["target_sequence"].apply(lambda x: d_rows_dict_pro[x])
    f5 = f5[["compound_iso_smiles", "BindingDB_reactant_id", "target_sequence", "affinity", "protein_id"]]

    f6 = pd.concat([f4, f5], axis=0)
    f6 = f6.reset_index(drop=True)

    f6.to_csv(outfile, sep="\t", index=False)
    print("The number of processed", target_name, "dataset:", len(f6))

    return f6, outfile

def kinase_dict(protein_name, dtype=None, kinase_file="/home/lydia/work/KinGen/data/kinase/kinase_info.tsv"):
    """
    Give kinase name and dtype, and return kinase KIBA ID, uniprot name or uniprot sequence etc.
    dtype can be "GENE", "SEQ", "KIBA_ID", "Uniprot_NAME".
    """
    kinase = pd.read_csv(kinase_file, sep="\t")
    kinase_dict = kinase.groupby(["Binding_NAME"]).apply(lambda x: x[["KIBA_ID","GENE","SEQ","Uniprot_NAME"]].to_dict("list")).to_dict()
    if dtype is not None:
        kinase_info = kinase_dict[protein_name][dtype][0]
        return kinase_info
    else:
        print("Require only 'GENE', 'KIBA_ID', 'SEQ' or 'Uniprot_NAME' as dtype!")


def scaffold_cluster(infile: str, outfile: str):
    """
    Fast clustering based on molecular backbones and randomly selecting one molecule from each cluster for output to a text file.
    """
    scaffolds = {}

    sdfloader = Chem.SmilesMolSupplier(infile, delimiter="\t", smilesColumn=1, nameColumn=2, titleLine=True)

    idx = 0
    for mol in sdfloader:
        if mol is  None:
            continue
        idx=idx+1
        smi=Chem.MolToSmiles(mol)
        if len(smi)==0:
            continue
        scaffold_smi =  MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        if scaffold_smi not in scaffolds.keys():
            scaffolds[scaffold_smi]=[smi]
        else:
            scaffolds[scaffold_smi].append(smi)
    print("Num of dataset:", idx, outfile)
    print("Num of Murcko scaffolds in dataset:",len(scaffolds.keys()))
    all_mols=[]
    for sca_smi in scaffolds.keys():
        mols=scaffolds[sca_smi]
        random.shuffle(mols)
        all_mols.append(mols[0])
    
    with open(outfile,"w") as out:
        for smi in all_mols:
            if smi is not None:
                out.write(smi+"\n")

# Scaffold split
def _generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
        return scaffold
    else:
        print("This molecule "+smiles+" cannot be transformed by rdikit.")
        pass
 
 
def generate_scaffolds(dataset: pd.DataFrame, log_every_n=1000):
    """
    Generate scaffolds and indexes in the given dataset.
    """
    scaffolds = {}
    non_scaffolds = [] # record smiles that can not be transformed by rdkit and generate scaffold.
    data_len = len(dataset)
    print(data_len)
 
    print("About to generate scaffolds")
    for ind, smiles in enumerate(dataset.compound_iso_smiles):
        if ind % log_every_n == 0:
            print("Generating scaffold %d/%d" % (ind, data_len))
        scaffold = _generate_scaffold(smiles)
        if scaffold is None:
            non_scaffolds.append(ind)
        else:
            if scaffold not in scaffolds:
                scaffolds[scaffold] = [ind]
            else:
                scaffolds[scaffold].append(ind)
 
    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    non_scaffolds = sorted(non_scaffolds)
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    return scaffold_sets, non_scaffolds
 
 
def scaffold_split(dataset:pd.DataFrame, test_size, valid_size=None, seed=None):

    scaffold_sets, non_scaffolds = generate_scaffolds(dataset)
    non_scaffold_inds = non_scaffolds
    train_inds: List[int] = []
    valid_inds: List[int] = []
    test_inds: List[int] = []

    if valid_size:
        train_size = 1.0 - valid_size - test_size
        train_cutoff = train_size * len(dataset)
        valid_cutoff = (train_size + valid_size) * len(dataset)
    
        print("About to sort in scaffold sets")
        random.seed(seed)
        random.shuffle(scaffold_sets)
        for scaffold_set in scaffold_sets:
            if len(train_inds) + len(scaffold_set) > train_cutoff:
                if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                    test_inds += scaffold_set
                else:
                    valid_inds += scaffold_set
            else:
                train_inds += scaffold_set
        return train_inds, valid_inds, test_inds, non_scaffold_inds
    else:
        train_size = 1.0 - test_size
        train_cutoff = train_size * len(dataset)
    
        print("About to sort in scaffold sets")
        random.seed(seed)
        random.shuffle(scaffold_sets)
        for scaffold_set in scaffold_sets:
            if len(train_inds) + len(scaffold_set) > train_cutoff:
                test_inds += scaffold_set
            else:
                train_inds += scaffold_set
        return train_inds, test_inds, non_scaffold_inds
    

def generate_dataset(dataset: pd.DataFrame, test_size, valid_size=None, kfolds=5, scaffold=False):
    data_train_list = []
    data_test_list = []
    data_valid_list = []

    if scaffold:
        # Here kfolds from the FusionDTA artical
        for i in range(kfolds):
            seed = (i+1)*23 # set by myself
            if valid_size:
                train_inds, valid_inds, test_inds, non_scaffold_inds = scaffold_split(dataset=dataset, test_size=test_size, valid_size=valid_size, seed=seed)
                data_valid = dataset.loc[valid_inds, :]
                data_valid = data_valid.reset_index(drop=True)
                data_valid_list.append(data_valid)
            else:
                train_inds, test_inds, non_scaffold_inds = scaffold_split(dataset=dataset, test_size=test_size, seed=seed)

            data_train = dataset.loc[train_inds, :]
            data_train = data_train.reset_index(drop=True)
            data_train_list.append(data_train)
            data_test = dataset.loc[test_inds, :]
            data_test = data_test.reset_index(drop=True)
            data_test_list.append(data_test)

        data_non_scaffolds = dataset.loc[non_scaffold_inds, :]
        data_non_scaffolds = data_non_scaffolds.reset_index(drop=True)    

        if valid_size:
            return data_train_list, data_test_list, data_valid_list, data_non_scaffolds
        else:
            return data_train_list, data_test_list, data_non_scaffolds
    else:
        data = dataset.sample(frac=1.0)
        rows, _ = data.shape
        split_index_1 = int(rows * test_size)
        if valid_size:
            split_index_2 = int(rows * (test_size + valid_size))
            data_test = data.iloc[0: split_index_1, :]
            data_valid = data.iloc[split_index_1:split_index_2, :]
            data_train = data.iloc[split_index_2: rows, :]
            return data_train, data_test, data_valid
        else:
            data_test = data.iloc[0: split_index_1, :]
            data_train = data.iloc[split_index_1: rows, :]
            return data_train, data_test


def extract_smiles(protein_name, dataset, store_path="/home/lydia/work/KinGen/data/results", dtype=None, is_gen=True):
    """
    Extract generated smiles from "memory.csv" and write smiles to "memory.smi" with "memory.sdf".
    ********NEW********
    Extract processed BindingDB bioactive ligands to "bindingdb.smi" with "bindingdb.sdf".
    """
    if is_gen:
        if dtype is None:
            smi_dir1 = os.path.join(store_path, protein_name, dataset)
            smi_dir = os.path.join(smi_dir1, os.listdir(smi_dir1)[-1])
            os.chdir(smi_dir)
            print("-- -- -- -- "*50, "\n")
            print("Working directory is", os.getcwd(), "\n")
            smi_df = pd.read_csv("memory.csv")
            smiles = list(smi_df["Smiles"])
            with open("memory.smi", "a") as F:
                for smi in smiles:
                    F.write(smi+"\n")
            writer = Chem.SDWriter("memory.sdf")
            for smi in smiles:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    writer.write(mol)
            writer.close()
            print("Finish extraction process of", protein_name, "in the", dataset.upper(), "dataset.\n")
            print("-- -- -- -- "*50, "\n\n")
        if dtype in ["fusion", "specific", "all"]:
            smi_dir1 = os.path.join(store_path, protein_name, dataset)
            if dtype == "fusion":
                smi_dir1 = os.path.join(smi_dir1, "fusion_dta")
            elif dtype == "specific":
                smi_dir1 = os.path.join(smi_dir1, protein_name)
            elif dtype == "all":
                smi_dir1 = os.path.join(smi_dir1, "17_kinases")
            if os.path.exists(smi_dir1):
                for dir in os.listdir(smi_dir1):
                    smi_dir = os.path.join(smi_dir1, dir)
                    if os.path.isdir(smi_dir):
                        os.chdir(smi_dir)
                        print("-- -- -- -- "*50, "\n")
                        print("Working directory is", os.getcwd(), "\n")
                        smi_df = pd.read_csv("memory.csv")
                        smiles = list(smi_df["Smiles"])
                        with open("memory.smi", "a") as F:
                            for smi in smiles:
                                F.write(smi+"\n")
                        writer = Chem.SDWriter("memory.sdf")
                        for smi in smiles:
                            mol = Chem.MolFromSmiles(smi)
                            if mol is not None:
                                writer.write(mol)
                        writer.close()
                        print("Finish extraction process of", protein_name, "in the", dataset.upper(), "dataset when dtype is", dtype.upper(), "\n")
                        print("-- -- -- -- "*50, "\n\n")
    else:
        smi_dir = os.path.join(store_path, protein_name)
        os.chdir(smi_dir)
        print("Working directory is", os.getcwd())
        pro_df = pd.read_csv("keepmean-norepeat_"+protein_name+".tsv", sep="\t")
        smiles = list(pro_df["compound_iso_smiles"])
        with open("bindingdb.smi", "a") as F:
            for smi in smiles:
                F.write(smi+"\n")

        writer = Chem.SDWriter("bindingdb.sdf")
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                writer.write(mol)

        writer.close()
        print("Finish extraction process of", protein_name, "from BindingDB.")


def cp_smiles_to_wd(protein_name, dataset, kingen_path="/home/lydia/work/KinGen", docking_dir_name="generated-mols", dtype=None):
    """
    Copy extracted smiles file to docking work directory.
    """
    if dtype is None:
        smi_dir1 = os.path.join(kingen_path, "data/results", protein_name, dataset)
        smi_dir = os.path.join(smi_dir1, os.listdir(smi_dir1)[-1])
        os.chdir(smi_dir)
        print("-- -- -- -- "*50, "\n")
        print("SMILES directory is", os.getcwd(), "\n")
        docking_dir = os.path.join(kingen_path, "evaluator/data", docking_dir_name, protein_name, "ligands")
        if not os.path.exists(docking_dir):
            os.mkdir(docking_dir)
        docking_filename = os.path.join(docking_dir, dataset + "_" + os.listdir(smi_dir1)[-1])
        shutil.copyfile("memory.smi", docking_filename+".smi")
        shutil.copyfile("memory.sdf", docking_filename+".sdf")
        print(os.listdir(docking_dir), "has been copied to the docking directory", docking_dir, "\n")
        print("-- -- -- -- "*50, "\n\n")
    if dtype in ["fusion", "specific", "all"]:
        smi_dir1 = os.path.join(kingen_path, "data/results", protein_name, dataset)
        if dtype == "fusion":
            smi_dir1 = os.path.join(smi_dir1, "fusion_dta")
        elif dtype == "specific":
            smi_dir1 = os.path.join(smi_dir1, protein_name)
        elif dtype == "all":
            smi_dir1 = os.path.join(smi_dir1, "17_kinases")
        if os.path.exists(smi_dir1):
            for dir in os.listdir(smi_dir1):
                smi_dir = os.path.join(smi_dir1, dir)
                if os.path.isdir(smi_dir):
                    os.chdir(smi_dir)
                    print("-- -- -- -- "*50, "\n")
                    print("SMILES directory is", os.getcwd(), "\n")
                    docking_dir = os.path.join(kingen_path, "evaluator/data", docking_dir_name, protein_name, "ligands")
                    if not os.path.exists(docking_dir):
                        os.mkdir(docking_dir)
                    docking_filename = os.path.join(docking_dir, dataset + "_" + dtype + "_" + dir)
                    shutil.copyfile("memory.smi", docking_filename+".smi")
                    shutil.copyfile("memory.sdf", docking_filename+".sdf")
                    print(os.listdir(docking_dir), "has been copied to the docking directory", docking_dir, "\n")
                    print("-- -- -- -- "*50, "\n\n")


def extract_glide_score(protein_name, dataset="davis", dtype="all",
                        store_path="../../evaluator/output/docking-output_generated",
                        kinase_info_path="kinase_info.tsv",
                        is_gen=True, return_df=False):
    """
    Extract glide score of generated molecules or BindingDB bioactive ligands.

    is_gen: whether is generated molecules or not.
    return_df: only return dataframe without saving as a csv file when is True.
    """
    protein_name = kinase_dict(protein_name=protein_name, dtype="GENE",
                               kinase_file=kinase_info_path).lower().split("/")[0]
    df_list = []

    if is_gen:
        # csv_file and sdf_file are from Glide docking.
        csv_file = store_path + "/*" + protein_name + "*" + dataset + "*" + dtype + "*csv"
        csv_file = glob.glob(csv_file)
        sdf_file = store_path + "/*" + protein_name + "*" + dataset + "*" + dtype + "*sdf"
        sdf_file = glob.glob(sdf_file)
        print("Find {} and {}".format(csv_file, sdf_file))
        
        if len(csv_file) > 0:
            for i in range(len(csv_file)):
                mols = Chem.SDMolSupplier(sdf_file[i])
                smis = []
                names = []
                for mol in mols:
                    smi = Chem.MolToSmiles(mol)
                    name = mol.GetProp("_Name")
                    smis.append(smi)
                    names.append(name)

                df = pd.DataFrame({"Title": names, "Smiles": smis})
                df["Docking Score"] = pd.read_csv(csv_file[i])["docking score"]
                if not return_df:
                    out_path = os.path.join(store_path, "gen_smi_docking")
                    if not os.path.exists(out_path):
                        os.mkdir(out_path)
                    out_path = os.path.join(out_path, "processed_"+csv_file[i].split("/")[-1])
                    df.to_csv(out_path, index=False)
                else:
                    df["Glide Resource"] = csv_file[i].split("/")[-1]
                    df_list.append(df)
        else:
            print("Do Not Find Target Files!")

    else:
        csv_file = store_path + "/*" + protein_name + "*csv"
        csv_file = glob.glob(csv_file)
        sdf_file = store_path + "/*" + protein_name + "*sdf"
        sdf_file = glob.glob(sdf_file)
        print("Find {} and {}".format(csv_file, sdf_file))

        if len(csv_file) > 0:
            for i in range(len(csv_file)):
                mols = Chem.SDMolSupplier(sdf_file[0])
                smis = []
                names = []
                for mol in mols:
                    smi = Chem.MolToSmiles(mol)
                    name = mol.GetProp("_Name")
                    smis.append(smi)
                    names.append(name)

                df = pd.DataFrame({"Title": names, "Smiles": smis})
                df["Docking Score"] = pd.read_csv(csv_file[i])["docking score"]
                if not return_df:
                    out_path = os.path.join(store_path, "bingdingdb_smi_docking")
                    if not os.path.exists(out_path):
                        os.mkdir(out_path)
                    out_path = os.path.join(out_path, "processed_"+csv_file[i].split("/")[-1])
                    df.to_csv(out_path, index=False)
                else:
                    df["Glide Resource"] = csv_file[i].split("/")[-1]
                    df_list.append(df)
        else:
            print("Do Not Find Target Files!")

    if return_df:
        return df_list


def map_glide_fusion_score(protein_name, dataset="davis", dtype="all", store_path="../results",
                           docking_store_path="../../evaluator/output/docking-output_generated",
                           return_df=False):
    """
    Map generated molecules' glide score and fusion score.
    Note:
    store_path is based on the generated molecules.
    """

    if dtype in ["fusion", "specific", "all"]:
        smi_dir1 = os.path.join(store_path, protein_name, dataset)
        if dtype == "fusion":
            smi_dir1 = os.path.join(smi_dir1, "fusion_dta")
        elif dtype == "specific":
            smi_dir1 = os.path.join(smi_dir1, protein_name)
        elif dtype == "all":
            smi_dir1 = os.path.join(smi_dir1, "17_kinases")
        if os.path.exists(smi_dir1):
            for dir in os.listdir(smi_dir1):
                smi_dir = os.path.join(smi_dir1, dir)
                if os.path.isdir(smi_dir):
                    os.chdir(smi_dir)
                    print("-- -- -- -- "*50, "\n")
                    print("Working directory is", os.getcwd(), "\n")
                    gen_df = pd.read_csv("memory.csv")
                    glide_df_list = extract_glide_score(protein_name, dataset=dataset, dtype=dtype, store_path=docking_store_path, return_df=True)
                    new_glide_df_list = []
                    for i in range(len(glide_df_list)):
                        glide_df = glide_df_list[i]
                        scores = []
                        smiles = []
                        for r in glide_df.index:
                            gid = glide_df.at[r, "Title"]
                            gid = int(gid.split(":")[-1]) - 1
                            score = gen_df.loc[gid,  "score"]
                            smi = gen_df.loc[gid, "Smiles"]
                            scores.append(score)
                            smiles.append(smi)
                        glide_df["Fusion Score"] = scores
                        glide_df["Generated Smiles"] = smiles
                        filename = glide_df["Glide Resource"][0]
                        glide_df = glide_df[["Title", "Smiles", "Docking Score", "Generated Smiles", "Fusion Score"]]
                        if not return_df:
                            out_path = os.path.join(docking_store_path, "gen_smi_docking+fusion")
                            if not os.path.exists(out_path):
                                os.mkdir(out_path)
                            out_path = os.path.join(out_path, "processed_"+filename)
                            glide_df.to_csv(out_path, index=False)
                        else:
                            new_glide_df_list.append(glide_df)
    if return_df:
        return new_glide_df_list


def cp_glide_score(protein_name, dataset="davis", dtype="all", store_path="/home/lydia/work/REINVENT/data/results", docking_store_path="/home/lydia/work/REINVENT/docking/output/docking-output_JAN", return_df=False):
    """
    Copy extracted information file of generated molecules' glide score to determined store path
    Note:
    store_path is based on the generated molecules.
    """

    if dtype in ["fusion", "specific", "all"]:
        smi_dir1 = os.path.join(store_path, protein_name, dataset)
        if dtype == "fusion":
            smi_dir1 = os.path.join(smi_dir1, "fusion_dta")
        elif dtype == "specific":
            smi_dir1 = os.path.join(smi_dir1, protein_name)
        elif dtype == "all":
            smi_dir1 = os.path.join(smi_dir1, "17_kinases")
        if os.path.exists(smi_dir1):
            for dir in os.listdir(smi_dir1):
                smi_dir = os.path.join(smi_dir1, dir)
                if os.path.isdir(smi_dir):
                    out_path = smi_dir
                    df_list = map_glide_fusion_score(protein_name, dataset=dataset, dtype=dtype,
                                                     store_path=store_path, docking_store_path=docking_store_path, return_df=True)
                    new_df_list = []
                    for i in range(len(df_list)):
                        df = df_list[i]
                        new_df = df[['Generated Smiles','Docking Score']]
                        new_df.columns = ['Smiles','score']
                        if not return_df:
                            out_path = os.path.join(out_path, "memory_docking_score.csv")
                            new_df.to_csv(out_path, index=False)
                        else:
                            new_df_list.append(new_df)
    if return_df:
        return new_df_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--kinase', default='ALK_TYROSINE_KINASE_RECEPTOR.tsv', help='kinase filename, you should put this file in the same location as kinase_process.py')
    parser.add_argument('--scaffold', action='store_true', help='split dataset by scaffolds or not')
    parser.add_argument('--kfolds', type=int, default=5, help='give kfolds when spliting dataset by scaffold')
    args = parser.parse_args()

    kinase_file = args.kinase
    is_scaffold = args.scaffold
    kfolds = args.kfolds
    
    print("Start Processing", kinase_file)
    dataframe, processed_outfile = keep_duplicate_mean(kinase_file)
    if is_scaffold:
        data_train_list, data_test_list, data_non_scaffolds = generate_dataset(dataset=dataframe, test_size=1/6, kfolds=kfolds, scaffold=is_scaffold)
    else:
        data_train, data_test = generate_dataset(dataset=dataframe, test_size=1/6)
    
    filename = kinase_file.split(".")[-2]
    if not os.path.exists(filename):
        os.mkdir(filename)
    shutil.copy(kinase_file, filename)
    if is_scaffold:
        for i in range(len(data_train_list)):
            train_outfile_path = os.path.join(filename, filename+"_train_fold"+str(i)+"_scaffold.csv")
            test_outfile_path = os.path.join(filename, filename+"_test_fold"+str(i)+"_scaffold.csv")
            data_train_list[i].to_csv(train_outfile_path)
            data_test_list[i].to_csv(test_outfile_path)
        non_scaffolds_outfile_path = os.path.join(filename, filename+"_non_scaffolds.csv")
        data_non_scaffolds.to_csv(non_scaffolds_outfile_path)
    else:
        train_outfile_path = os.path.join(filename, filename+"_train_random.csv")
        test_outfile_path = os.path.join(filename, filename+"_test_random.csv")
        data_train.to_csv(train_outfile_path)
        data_test.to_csv(test_outfile_path)