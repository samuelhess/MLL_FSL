import numpy as np
import random
import numpy.random as rng
import pickle
from utils.utils import parse_args
from utils.metrics import mll_dist


if __name__=='__main__':
    params = parse_args()
    print('Running MLL Few Shot Inductive Evaluation on Preprocessed Features...')
    print(vars(params))

    n_shot = params.n_shot
    mll_thresh = params.mll_thresh
    data_type = 'test'
    rng.seed()
    network_type = params.network
    dataset = params.dataset
    n_processors = params.n_processors
    n_way = params.n_way
    n_query = params.n_query
    total_query = n_query * n_way
    n_support = n_shot
    n_samples = params.n_samples
    n_metrics = 3

    # Create ground truth labels
    gt = list(np.repeat(np.arange(n_way), n_query))

    # Set paths
    preprocess_feature_path = '../data/'
    base_path = preprocess_feature_path + dataset + '/' + network_type + '/preprocessed_features/' + dataset + '_' + network_type
    group_statistics_file = base_path  + f'_{n_shot}_' + 'group_val_stats.json'

    # Filenames
    novel_file_mll = base_path + '_mll_' + data_type +'.plk'

    # Load preprocessed features
    cl_data_file_mll = pickle.load( open( novel_file_mll, "rb" ) )

    # Get class list
    class_list = cl_data_file_mll.keys()

    #matrix initialization
    all_mll_final = np.zeros(n_samples)

    # Get metrics for each test
    for i_sample in range(n_samples):

        # Get random classes
        select_class = random.sample(class_list,n_way)
        z_all_mll  = []

        # Get random support and query samples for each class
        for cl in select_class:
            img_feat_mll = np.array(cl_data_file_mll[cl],dtype='float32')
            perm_ids = np.random.permutation(len(img_feat_mll)).tolist()
            z_all_mll.append( [ np.squeeze( img_feat_mll[perm_ids[i]]) for i in range(n_support+n_query) ] )     # stack each batch

        # All samples
        z_all_mll = np.array(z_all_mll)
        z_all_mll = z_all_mll/np.linalg.norm(z_all_mll,ord=2,axis=1,keepdims=1)

        # Query samples
        z_query_mll = z_all_mll[:,n_support:,:].reshape(n_query*n_way,-1)

        # Support Samples
        z_support_mll = z_all_mll[:,:n_support,:].mean(axis=1)
        
        # Compute metrics

        pred_mll, scores_mll = mll_dist(z_support_mll,z_query_mll,mll_thresh)
        acc_mll = np.sum(pred_mll == gt)/np.shape(pred_mll)[0]
        all_mll_final[i_sample] = acc_mll

        
    print('********** Test Results **********')
    print(f'MLL Acc: {all_mll_final.mean()*100:.4f} +/- {all_mll_final.std()/np.sqrt(n_samples)*100:.4f}') 
    print('***********************************')       
    

