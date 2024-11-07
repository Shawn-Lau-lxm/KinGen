import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from sklearn.model_selection import KFold
from src.getdata import getdata_from_csv
from src.utils import DrugTargetDataset, collate, AminoAcid, ci
from src.models.DAT import DAT3
import os

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', default=True, help='Disables CUDA training.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--batchsize', type=int, default=256, help='Number of batch_size')
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
parser.add_argument('--training-dataset-path', default='data/kiba_train.csv', help='training dataset path: davis or kiba/ 5-fold or not')
parser.add_argument('--testing-dataset-path', default='data/kiba_test.csv', help='training dataset path: davis or kiba/ 5-fold or not')
parser.add_argument('--kinase', default='ALK_TYROSINE_KINASE_RECEPTOR', help='kinase name')
parser.add_argument('--fold', type=int, default=5, help='k for k-fold cross validation training')
parser.add_argument('--cuda-visble-devices', default='0', help='set cuda device number')
parser.add_argument('--scaffold', action='store_true', help='split dataset by scaffolds or not')
parser.add_argument('--kfold-cv', action='store_false', help='perform k-fold cross validation or not')
parser.add_argument('--early-stop', type=int, default=100, help='early stop steps')
parser.add_argument('--whole-set', action='store_true', help='use the entire dataset to train the model by cross validation,leading to different log_dir')

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
early_stop = args.early_stop
is_whole_set = args.whole_set

# Just for debug
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
# kinase = "EPIDERMAL_GROWTH_FACTOR_RECEPTOR"
# training_dataset_address = "../data/kinase/EPIDERMAL_GROWTH_FACTOR_RECEPTOR/EPIDERMAL_GROWTH_FACTOR_RECEPTOR_train_random.csv"
# testing_dataset_address = "../data/kinase/EPIDERMAL_GROWTH_FACTOR_RECEPTOR/EPIDERMAL_GROWTH_FACTOR_RECEPTOR_test_random.csv"

#processing training data and validating data
if is_pretrain:
    train_drug, train_protein, train_affinity, pid = getdata_from_csv(training_dataset_address, maxlen=1536)

else:
    train_drug, train_protein, train_affinity = getdata_from_csv(training_dataset_address, maxlen=1024)
    train_protein = [x.encode('utf-8').upper() for x in train_protein]
    train_protein = [torch.from_numpy(Alphabet.encode(x)).long() for x in train_protein]
train_affinity = torch.from_numpy(np.array(train_affinity)).float()

dataset_train = DrugTargetDataset(train_drug, train_protein, train_affinity, pid, is_target_pretrain=is_pretrain, self_link=False,dataset=dataset)

if is_kfold_cv:
    # Add kFold cross validation process
    kf = KFold(n_splits=int(fold), shuffle=True, random_state=123)
    dataloader_train_list = []
    dataloader_valid_list = []
    for train, valid in kf.split(range(len(dataset_train.Y))):
        train_sampler = torch.utils.data.SubsetRandomSampler(train)
        valid_sampler = torch.utils.data.SubsetRandomSampler(valid)
        dataloader_train = torch.utils.data.DataLoader(dataset_train
                                                    , batch_size=batch_size
                                                    , sampler = train_sampler
                                                    , collate_fn=collate
                                                    #, drop_last=True # only for egfr
                                                    )
        dataloader_valid = torch.utils.data.DataLoader(dataset_train
                                                    , batch_size=batch_size
                                                    , sampler=valid_sampler
                                                    , collate_fn=collate
                                                    )
        dataloader_train_list.append(dataloader_train)
        dataloader_valid_list.append(dataloader_valid)
else:
    dataloader_train = torch.utils.data.DataLoader(dataset_train
                                                    , batch_size=batch_size
                                                    , shuffle=True
                                                    , collate_fn=collate
                                                    )
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
                                                    , shuffle=True
                                                    , collate_fn=collate
                                                    )

if is_kfold_cv:
    for k in range(len(dataloader_train_list)):
        #model
        model = DAT3(embedding_dim, rnn_dim, hidden_dim, graph_dim, dropout, alpha, n_heads, is_pretrain=is_pretrain)
        model.load_state_dict(torch.load('saved_models/DAT_best_'+dataset+'_65smiles.pkl')['model'], strict=False)


        if use_cuda:
            model.cuda()


        #optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optim = torch.optim.Adam(params, lr=lr)
        criterion = nn.MSELoss()

        dataloader_train = dataloader_train_list[k]
        dataloader_valid = dataloader_valid_list[k]
        train_epoch_size = len(train_sampler)
        valid_epoch_size =len(valid_sampler)

        print('--'*50 + ' GAT model-FOLD ' + str(k) + '--'*50)

        best_ci = 0
        best_mse = 100000

        if is_whole_set:
            log_dir = os.path.join("saved_models", kinase, dataset, "whole_set_cv")
        else:
            log_dir = os.path.join("saved_models", kinase, dataset)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if is_scaffold:
            log_name = os.path.join(log_dir,"DAT_best_"+dataset+"_65smiles-scaffold-"+str(fold)+"fold"+str(k))
        else:
            log_name = os.path.join(log_dir,"DAT_best_"+dataset+"_65smiles-random-"+str(fold)+"fold"+str(k))
        log_filepath = log_name + ".mse.ci"
        f_log = open(log_filepath, 'w')

        for epoch in range(epochs):

            #train
            model.train()
            b = 0
            total_loss = []
            total_ci = []

            for protein, smiles, affinity in dataloader_train:

                if use_cuda:
                    protein = [p.cuda() for p in protein]
                    smiles = [s.cuda() for s in smiles]
                    affinity = affinity.cuda()

                _, out = model(protein, smiles)
                loss = criterion(out, affinity)

                loss.backward()
                optim.step()
                optim.zero_grad()

                out = out.cpu()
                affinity = affinity.cpu()
                loss = loss.cpu().detach()
                c_index = ci(affinity.detach().numpy(),out.detach().numpy())

                b = b + batch_size
                total_loss.append(loss)
                total_ci.append(c_index)
                print('# [{}/{}] training {:.1%} loss={:.5f}, ci={:.5f}\n'.format(epoch+1
                                                                            , epochs
                                                                            , b/train_epoch_size
                                                                            , loss 
                                                                            , c_index

                             , end='\r'))

            print('Train total_loss={:.5f}, total_ci={:.5f}\n'.format(np.mean(total_loss), np.mean(total_ci)))
            f_log.write('Train\tEpoch{}\t{:.5f}\t{:.5f}\n'.format(epoch, np.mean(total_loss), np.mean(total_ci)))

            model.eval()
            b=0
            total_loss = []
            total_ci = []
            total_pred = torch.Tensor()
            total_label = torch.Tensor()
            with torch.no_grad():
                for protein, smiles, affinity in dataloader_valid:
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

                    print('# [{}/{}] validating {:.1%} loss={:.5f}, ci={:.5f}\n'.format(epoch+1
                                                                                , epochs
                                                                                , b/valid_epoch_size
                                                                                , loss 
                                                                                , c_index
                                                                                )
                    , end='\r')



            all_ci = ci(total_label.detach().numpy().flatten(),total_pred.detach().numpy().flatten())

            print('Valid total_loss={:.5f}, total_ci={:.5f}\n'.format(np.mean(total_loss), all_ci))
            f_log.write('Valid\tEpoch{}\t{:.5f}\t{:.5f}\n'.format(epoch, np.mean(total_loss), all_ci))
            f_log.flush()

            save_path = log_name +".pkl"
            if all_ci > best_ci:
                best_ci = all_ci
            if np.mean(total_loss) < best_mse:
                best_mse = np.mean(total_loss)
                save_dict = {'model':model.state_dict(), 'optim':optim.state_dict()}
                torch.save(save_dict, save_path)
                print('Saving model with loss {:.3f}...\n'.format(np.mean(total_loss)))
                early_stop_count = 0
            else:
                early_stop_count += 1
            if early_stop_count >= early_stop:
                print('\nModel is not improving, so we halt the training session.')
                break

        f_log.close()

else:
    #model
    model = DAT3(embedding_dim, rnn_dim, hidden_dim, graph_dim, dropout, alpha, n_heads, is_pretrain=is_pretrain)
    model.load_state_dict(torch.load('saved_models/DAT_best_'+dataset+'_65smiles.pkl')['model'], strict=False)


    if use_cuda:
        model.cuda()


    #optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.Adam(params, lr=lr)
    criterion = nn.MSELoss()

    train_epoch_size = len(train_drug)
    test_epoch_size = len(test_drug)
    print('--'*50 + ' GAT model ' + '--'*50)
    best_ci = 0
    best_mse = 100000
    log_dir = os.path.join("saved_models", kinase, dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if is_scaffold:
        log_name = os.path.join(log_dir,"DAT_best_"+dataset+"_65smiles-scaffold")
    else:
        log_name = os.path.join(log_dir,"DAT_best_"+dataset+"_65smiles-random")
    log_filepath = log_name + ".mse.ci"
    f_log = open(log_filepath, 'w')
    for epoch in range(epochs):
        #train
        model.train()
        b = 0
        total_loss = []
        total_ci = []
        for protein, smiles, affinity in dataloader_train:
            if use_cuda:
                protein = [p.cuda() for p in protein]
                smiles = [s.cuda() for s in smiles]
                affinity = affinity.cuda()
            _, out = model(protein, smiles)
            loss = criterion(out, affinity)
            loss.backward()
            optim.step()
            optim.zero_grad()
            out = out.cpu()
            affinity = affinity.cpu()
            loss = loss.cpu().detach()
            c_index = ci(affinity.detach().numpy(),out.detach().numpy())
            b = b + batch_size
            total_loss.append(loss)
            total_ci.append(c_index)
            print('# [{}/{}] training {:.1%} loss={:.5f}, ci={:.5f}\n'.format(epoch+1
                                                                        , epochs
                                                                        , b/train_epoch_size
                                                                        , loss 
                                                                        , c_index
                         , end='\r'))
        print('Train total_loss={:.5f}, total_ci={:.5f}\n'.format(np.mean(total_loss), np.mean(total_ci)))
        f_log.write('Train\tEpoch{}\t{:.5f}\t{:.5f}\n'.format(epoch, np.mean(total_loss), np.mean(total_ci)))
        model.eval()
        b=0
        total_loss = []
        total_ci = []
        total_pred = torch.Tensor()
        total_label = torch.Tensor()
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
                print('# [{}/{}] testing {:.1%} loss={:.5f}, ci={:.5f}\n'.format(epoch+1
                                                                            , epochs
                                                                            , b/test_epoch_size
                                                                            , loss 
                                                                            , c_index
                                                                            )
                , end='\r')
        all_ci = ci(total_label.detach().numpy().flatten(),total_pred.detach().numpy().flatten())
        print('Test total_loss={:.5f}, total_ci={:.5f}\n'.format(np.mean(total_loss), all_ci))
        f_log.write('Test\tEpoch{}\t{:.5f}\t{:.5f}\n'.format(epoch, np.mean(total_loss), all_ci))
        f_log.flush()
        save_path = log_name +".pkl"
        if all_ci > best_ci:
            best_ci = all_ci
        if np.mean(total_loss) < best_mse:
            best_mse = np.mean(total_loss)
            save_dict = {'model':model.state_dict(), 'optim':optim.state_dict()}
            torch.save(save_dict, save_path)
            print('Saving model with loss {:.3f}...\n'.format(np.mean(total_loss)))
            early_stop_count = 0
        else:
            early_stop_count += 1
        if early_stop_count >= early_stop:
            print('\nModel is not improving, so we halt the training session.')
            break
        
    f_log.close()





