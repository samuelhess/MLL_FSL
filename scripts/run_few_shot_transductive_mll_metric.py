
import numpy as np
import numpy.random as rng
import pickle
from utils.dataset import SimpleDataset
from utils.sampler import CategoriesSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.utils import parse_args
from utils.metrics import mll_transductive

if __name__=='__main__':
    rng.seed()
    params = parse_args('transductive')
    print('Running MLL Few Shot Transductive Evaluation on Preprocessed Features...')
    print(vars(params))

    if params.balanced:
        distribution_type = 'balanced'
    else:
        distribution_type = 'dirichlet'
    dirichlet_alpha = params.dirichlet_alpha
    
    #Set up/load parameters
    n_shot = params.n_shot
    mll_thresh = params.mll_thresh
    data_type = 'test'
    network_type = params.network
    dataset_name = params.dataset
    n_way = params.n_way
    n_query = params.n_query
    total_query = n_query * n_way
    n_support = n_shot
    n_samples = params.n_samples

    clip_val = params.mll_thresh
    iterations = params.iterations
    alpha_c = params.alpha
    total_query = n_query * n_way
    
    # Create ground truth labels
    gt = list(np.repeat(np.arange(n_way), n_query))

    # Set paths
    preprocess_feature_path = '../data/'
    base_path = preprocess_feature_path + dataset_name + '/' + network_type + '/preprocessed_features/' + dataset_name + '_' + network_type

    # Filename
    novel_file_mll = base_path + '_mll_' + data_type +'.plk'

    # Load preprocessed features
    data = pickle.load( open( novel_file_mll, "rb" ) )

    # get number of features from the data
    n_features = np.shape(data[0])[1]

    # Setup loader with 'balanced' or 'dirichlet' distribution: defaults to 'dirichlet'
    dataset = SimpleDataset(novel_file_mll)
    if dataset_name == 'miniImageNet':
        dataset.n_samples_per_label = 600
    elif dataset_name == 'tieredImageNet':
        dataset.n_samples_per_label = 900
    else:
        dataset.n_samples_per_label = dataset.n_samples_per_label

    all_labels = dataset.get_labels()
    sampler = CategoriesSampler(label=all_labels, n_batch=n_samples, n_cls=n_way, s_shot=n_shot, q_shot=n_query, balanced=distribution_type, alpha=dirichlet_alpha)
    novel_loader = DataLoader(dataset, batch_sampler=sampler, pin_memory=True)

    #matrix initialization
    acc_mll_total = np.zeros(n_samples)
    z_support_mll = np.zeros((n_way,n_features))

    # Get metrics for each test
    for i_sample, batch in tqdm(enumerate(novel_loader, 0)):
        #load data and fix labels to be 0-n_way
        data = batch[0]
        labels = batch[1]
        labels_temp = np.copy(labels)
        for i_way in range(n_way):
            labels_temp[labels[i_way]==labels] = i_way
        
        #setup numpy matricies for labels,query and support
        labels = np.copy(labels_temp)
        z_all_mll = np.array(data)
        z_all_mll = z_all_mll/np.linalg.norm(z_all_mll,ord=2,axis=1,keepdims=1)
        # z_all_mll[z_all_mll > clip_val] = clip_val # done in metric function
        z_query_mll = np.copy(z_all_mll[n_support*n_way:,:])
        query_labels = np.copy(labels[n_support*n_way:])
        support_labels = np.copy(labels[:n_support*n_way])

        # Average all support features of a class
        for ic in range(n_way):
            support_ic = np.copy(z_all_mll[np.where(support_labels == ic)[0]])
            # temp[temp > clip_val] = clip_val # done in metric function
            z_support_mll[ic,:] = support_ic.mean(0)

        # run transductive iterations
        pred_mll, scores_mll = mll_transductive(z_support_mll,n_way,z_query_mll,clip_val,iterations,alpha_c)

        #compute accuracy
        acc_mll = np.sum(pred_mll == query_labels)/np.shape(pred_mll)[0]
        acc_mll_total[i_sample] = acc_mll


    print(acc_mll_total.mean())




