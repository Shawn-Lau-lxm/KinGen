import argparse
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.utils.data
from sklearn.model_selection import KFold
from src.getdata import getdata_from_csv
from src.utils import DrugTargetDataset, collate, AminoAcid, ci, r_squared_error
from src.models.DAT import DAT3

def evaluate_model(model: nn.Module, 
                   model_name: str, 
                   test_epoch_size: int, 
                   dataloader_test: torch.utils.data.dataloader, 
                   use_cuda: bool=True, 
                   use_valid: bool=True, 
                   batch_size: int=128):
    if use_cuda:
        model.cuda()
      
    #optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.Adam(params, lr=lr)
    criterion = nn.MSELoss()

    print('--- GAT model --- ')

    best_ci = 0

    model.eval()
    b=0
    total_loss = []
    total_ci = []
    total_pred = torch.Tensor()
    total_label = torch.Tensor()

    if use_valid:
        out_file = model_name.split(".")[0]+"_validataion-set.log"
    else:
        out_file = model_name.split(".")[0]+"_test-set.log"
    with torch.no_grad():
        for protein, smiles, affinity in dataloader_test:
            if use_cuda:
                protein = [p.cuda() for p in protein]
                smiles = [s.cuda() for s in smiles]
                affinity = affinity.cuda()

            _, out = model(protein, smiles)

            loss = criterion(out, affinity)

            out = out.cpu()
            affinity = affinity.cpu()
            loss = loss.cpu().detach()
            c_index = ci(affinity.detach().numpy(),out.detach().numpy())


            b = b + batch_size
            total_loss.append(loss)
            total_ci.append(c_index)
            total_pred = torch.cat((total_pred, out), 0)
            total_label = torch.cat((total_label, affinity), 0)

            print('# testing {:.1%} loss={:.5f}, ci={:.5f}\n'.format(
                                                                      b/test_epoch_size
                                                                    , loss 
                                                                    , c_index
                                                                    )
            , end='\r')

            with open(out_file, "a") as f:
                f.write('# testing {:.1%} loss={:.5f}, ci={:.5f}, protein={}, smiles={}, affinity={}\n'.format(b/test_epoch_size, loss, c_index, protein, smiles, affinity))
                f.flush()

    all_ci = ci(total_label.detach().numpy().flatten(),total_pred.detach().numpy().flatten())
    print(kinase, dataset, out_file)
    print('loss={:.5f}, ci={:.5f}\n'.format(np.mean(total_loss), all_ci))

    with open(out_file, "a") as f:
        f.write('loss={:.5f}, ci={:.5f}\n'.format(np.mean(total_loss), all_ci))
        f.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=True, help='Disables CUDA training.')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--batchsize', type=int, default=128, help='Number of batch_size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--embedding-dim', type=int, default=1280, help='dimension of embedding (default: 512)')
    parser.add_argument('--rnn-dim', type=int, default=128, help='hidden unit/s of RNNs (default: 256)')
    parser.add_argument('--hidden-dim', type=int, default=256, help='hidden units of FC layers (default: 256)')
    parser.add_argument('--graph-dim', type=int, default=256, help='Number of hidden units.')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of head attentions.')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--pretrain', action='store_false', help='protein pretrained or not')
    parser.add_argument('--dataset', default='davis', help='dataset: davis or kiba')
    parser.add_argument('--training-dataset-path', default='data/davis_train.csv', help='training dataset path: davis or kiba/ 5-fold or not')
    parser.add_argument('--testing-dataset-path', default='data/davis_test.csv', help='training dataset path: davis or kiba/ 5-fold or not')

    parser.add_argument('--kinase', default='ALK_TYROSINE_KINASE_RECEPTOR', help='kinase name')
    parser.add_argument('--fold', type=int, default=5, help='5-fold training')
    parser.add_argument('--cuda-visble-devices', default='0', help='set cuda device number')
    parser.add_argument('--scaffold', action='store_true', help='split dataset by scaffolds or not')
    parser.add_argument('--kfold-cv', action='store_false', help='perform k-fold cross validation or not')
    parser.add_argument('--use-valid', action='store_false', help='evaluate based on the cross validation set')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visble_devices

    dataset = args.dataset
    use_cuda = args.cuda and torch.cuda.is_available()

    batch_size = args.batchsize
    epochs = args.epochs
    lr = args.lr
    weight_decay = args.weight_decay

    embedding_dim = args.embedding_dim
    rnn_dim = args.rnn_dim
    hidden_dim = args.hidden_dim
    graph_dim = args.graph_dim

    n_heads = args.n_heads
    dropout = args.dropout
    alpha = args.alpha

    is_pretrain = args.pretrain

    Alphabet = AminoAcid()

    training_dataset_address = args.training_dataset_path
    testing_dataset_address = args.testing_dataset_path
    kinase = args.kinase
    fold = args.fold
    is_scaffold = args.scaffold
    is_kfold_cv = args.kfold_cv
    use_valid = args.use_valid

    if use_valid:
        #processing training data and validating data
        if is_pretrain:
            train_drug, train_protein, train_affinity, pid = getdata_from_csv(training_dataset_address, maxlen=1536)

        else:
            train_drug, train_protein, train_affinity = getdata_from_csv(training_dataset_address, maxlen=1024)
            train_protein = [x.encode('utf-8').upper() for x in train_protein]
            train_protein = [torch.from_numpy(Alphabet.encode(x)).long() for x in train_protein]
        train_affinity = torch.from_numpy(np.array(train_affinity)).float()

        dataset_train = DrugTargetDataset(train_drug, train_protein, train_affinity, pid, is_target_pretrain=is_pretrain, self_link=False,dataset=dataset)

        # Extract kFold cross validation sets list
        kf = KFold(n_splits=int(fold), shuffle=True, random_state=123)
        dataloader_valid_list = []
        valid_epoch_size_list = []
        for train, valid in kf.split(range(len(dataset_train.Y))):
            valid_sampler = torch.utils.data.SubsetRandomSampler(valid)
            dataloader_valid = torch.utils.data.DataLoader(dataset_train
                                                        , batch_size=batch_size
                                                        , sampler=valid_sampler
                                                        , collate_fn=collate
                                                        #, drop_last=True # only for egfr
                                                        )
            dataloader_valid_list.append(dataloader_valid)
            valid_epoch_size_list.append(len(valid_sampler))
    else:
        #processing testing data
        if is_pretrain:
            test_drug, test_protein, test_affinity, pid = getdata_from_csv(testing_dataset_address, maxlen=1536)
        else:
            test_drug, test_protein, test_affinity = getdata_from_csv(testing_dataset_address, maxlen=1024)
            test_protein = [x.encode('utf-8').upper() for x in test_protein]
            test_protein = [torch.from_numpy(Alphabet.encode(x)).long() for x in test_protein]
        test_affinity = torch.from_numpy(np.array(test_affinity)).float()

        dataset_test = DrugTargetDataset(test_drug, test_protein, test_affinity, pid, is_target_pretrain=is_pretrain, self_link=False,dataset=dataset)
        dataloader_test = torch.utils.data.DataLoader(dataset_test
                                                        , batch_size=batch_size
                                                        , shuffle=False
                                                        , collate_fn=collate
                                                        #, drop_last=True # True only for egfr
                                                        )
        test_epoch_size = len(test_drug)
    #model
    model = DAT3(embedding_dim, rnn_dim, hidden_dim, graph_dim, dropout, alpha, n_heads, is_pretrain=is_pretrain)

    model_dir = os.path.join("saved_models", kinase, dataset)
    if is_kfold_cv:
        for k in range(fold):
            if is_scaffold:
                model_name = os.path.join(model_dir,"DAT_best_"+dataset+"_65smiles-scaffold-"+str(fold)+"fold"+str(k)+".pkl")
            else:
                model_name = os.path.join(model_dir,"DAT_best_"+dataset+"_65smiles-random-"+str(fold)+"fold"+str(k)+".pkl")
            model.load_state_dict(torch.load(model_name)['model'], strict=False)
            if use_valid:
                evaluate_model(model=model, model_name=model_name, test_epoch_size=valid_epoch_size_list[k], dataloader_test=dataloader_valid_list[k], use_cuda=use_cuda, use_valid=use_valid, batch_size=batch_size)
            else:
                evaluate_model(model=model, model_name=model_name, test_epoch_size=test_epoch_size, dataloader_test=dataloader_test, use_cuda=use_cuda, use_valid=use_valid, batch_size=batch_size)
    else:
        if is_scaffold:
            model_name = os.path.join(model_dir,"DAT_best_"+dataset+"_65smiles-scaffold"+".pkl")
        else:
            model_name = os.path.join(model_dir,"DAT_best_"+dataset+"_65smiles-random"+".pkl")
        #model_name = os.path.join(model_dir,"DAT_best_"+dataset+"_65smiles.pkl")
        model.load_state_dict(torch.load(model_name)['model'], strict=False)
        evaluate_model(model=model, model_name=model_name, use_cuda=use_cuda, test_epoch_size=test_epoch_size, dataloader_test=dataloader_test, use_valid=use_valid, batch_size=batch_size)
