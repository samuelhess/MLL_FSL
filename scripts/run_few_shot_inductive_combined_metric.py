import numpy as np
import random
import numpy.random as rng
import pickle
from scipy.stats import multivariate_normal
from multiprocessing import Pool
from utils.utils import parse_args, compute_and_save_statistics, load_statistics
from utils.metrics import euc_dist, mll_dist, cos_dist
import json


if __name__=='__main__':
    rng.seed()
    params = parse_args('inductive')
    print('Running Euclidean + Cosine + MLL Combined Few Shot Inductive Evaluation on Preprocessed Features...')
    print(vars(params))

    #Set up/load parameters
    compute_cov = False
    if params.compute_cov:
        compute_cov = True
    n_shot = params.n_shot
    mll_thresh = params.mll_thresh
    if compute_cov:
        data_type = 'val'
    else:
        data_type = 'test'
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
    novel_file_euc = base_path + '_euc_' + data_type +'.plk'
    novel_file_cos = base_path + '_cos_' + data_type +'.plk'

    # Load preprocessed features
    cl_data_file_mll = pickle.load( open( novel_file_mll, "rb" ) )
    cl_data_file_euc = pickle.load( open( novel_file_euc, "rb" ) )
    cl_data_file_cos = pickle.load( open( novel_file_cos, "rb" ) )

    # Get class list
    class_list = cl_data_file_mll.keys()

    # If validation compute statistics, if evaluation load them
    if not compute_cov:
        [cov_inclass, u_inclass, cov_crossclass, u_crossclass] = load_statistics(group_statistics_file)
        
    #This are just supporting functions to help the parallel processing
    # multivariate_normal across all the samples
    def f1(x):
        y = np.transpose(x,(1,2,0)).reshape((n_way*total_query,n_metrics))
        probs_inclass = multivariate_normal.cdf(y, mean=u_inclass, cov=cov_inclass)
        return probs_inclass
        
    def f2(x):
        y = np.transpose(x,(1,2,0)).reshape((n_way*total_query,n_metrics))
        probs_crossclass = 1-multivariate_normal.cdf(y, mean=u_crossclass, cov=cov_crossclass)
        return probs_crossclass

    #matrix initialization
    all_euc_acc = np.zeros(n_samples)
    all_cos_acc = np.zeros(n_samples)
    all_mll_acc = np.zeros(n_samples)
    all_combined_acc = np.zeros(n_samples)
    all_scores = np.zeros((n_samples,n_metrics,n_way,total_query))
    all_gt = np.zeros((n_samples,total_query))

    # Get all three metrics for each test
    for i_sample in range(n_samples):

        # Get random classes
        select_class = random.sample(class_list,n_way)
        z_all_mll  = []
        z_all_cos  = []
        z_all_euc  = []

        # Get random support and query samples for each class
        for cl in select_class:
            img_feat_mll = np.array(cl_data_file_mll[cl],dtype='float32')
            img_feat_euc = np.array(cl_data_file_euc[cl],dtype='float32')
            img_feat_cos = np.array(cl_data_file_cos[cl],dtype='float32')
            perm_ids = np.random.permutation(len(img_feat_mll)).tolist()
            z_all_mll.append( [ np.squeeze( img_feat_mll[perm_ids[i]]) for i in range(n_support+n_query) ] )     # stack each batch
            z_all_euc.append( [ np.squeeze( img_feat_euc[perm_ids[i]]) for i in range(n_support+n_query) ] )     # stack each batch
            z_all_cos.append( [ np.squeeze( img_feat_cos[perm_ids[i]]) for i in range(n_support+n_query) ] )     # stack each batch

        # All samples
        z_all_mll = np.array(z_all_mll)
        z_all_mll = z_all_mll/np.linalg.norm(z_all_mll,ord=2,axis=2,keepdims=1)
        z_all_mll[z_all_mll > mll_thresh] = mll_thresh
        z_all_cos = np.array(z_all_cos)
        z_all_cos = z_all_cos/np.linalg.norm(z_all_cos,ord=2,axis=2,keepdims=1)
        z_all_euc = np.array(z_all_euc)
        z_all_euc = z_all_euc/np.linalg.norm(z_all_euc,ord=2,axis=2,keepdims=1)


        # Query samples
        z_query_mll = z_all_mll[:,n_support:,:].reshape(n_query*n_way,-1)
        z_query_euc = z_all_euc[:,n_support:,:].reshape(n_query*n_way,-1)
        z_query_cos = z_all_cos[:,n_support:,:].reshape(n_query*n_way,-1)

        # Support Samples
        z_support_mll = z_all_mll[:,:n_support,:].mean(axis=1)
        z_support_euc = z_all_euc[:,:n_support,:].mean(axis=1)
        z_support_cos = z_all_cos[:,:n_support,:].mean(axis=1)
        
        # Compute metrics
        pred_euc, scores_euc = euc_dist(z_support_euc,z_query_euc)
        acc_euc = np.sum(pred_euc == gt)/np.shape(pred_euc)[0]

        pred_cos, scores_cos = cos_dist(z_support_cos,z_query_cos)
        acc_cos = np.sum(pred_cos == gt)/np.shape(pred_cos)[0]

        pred_mll, scores_mll = mll_dist(z_support_mll,z_query_mll,mll_thresh)
        acc_mll = np.sum(pred_mll == gt)/np.shape(pred_mll)[0]
        all_mll_acc[i_sample] = acc_mll

        # collect all scores
        all_scores[i_sample] = [scores_euc, scores_cos, scores_mll]
        all_gt[i_sample] = gt

    # If evaluation compute combined scores
    if not compute_cov:
        # Compute in-class probability
        with Pool(n_processors) as p:
            probs_inclass = p.map(f1, all_scores)

        # Compute cross-class probability
        with Pool(n_processors) as p:
            probs_crossclass = p.map(f2, all_scores)
            
        for i_sample in range(n_samples):
            # combined_metric = np.copy(probs_inclass[i_sample] - probs_crossclass[i_sample])
            combined_metric = np.copy(-np.sqrt((1-probs_inclass[i_sample])**2 + (probs_crossclass[i_sample])**2))
            # combined_metric = np.copy(probs_inclass[i_sample]*(1-probs_crossclass[i_sample]))
            combined_metric = combined_metric.reshape((n_way,total_query))
            pred_combined = np.argmax(combined_metric,axis=0)  
            combined_acc = np.sum(pred_combined == gt)/np.shape(pred_combined)[0]

            all_combined_acc[i_sample] = combined_acc
        

    # If validation data, compute/save statistics
    if compute_cov:
        inclass_features = []
        crossclass_features = []

        for idx,i_trial in enumerate(all_gt):
            for idx2,i_sample in enumerate(i_trial):
                inclass_features.append(all_scores[idx,:,int(i_sample),idx2]) # get the true class feature
                crossclass_features.append(all_scores[idx,:,int(np.mod(i_sample+1+np.random.choice(3),5)),idx2]) # get any other wrong class feature

        compute_and_save_statistics(inclass_features,crossclass_features,group_statistics_file)
        print('***********************************')       
        print('Saved Validation Statistics')
        print(f'MLL Validation Acc: {all_mll_acc.mean()*100:.4f} +/- {all_mll_acc.std()/np.sqrt(n_samples)*100:.4f}')
        print('***********************************')
    else: #print results
        print('********** Test Results **********')
        print(f'MLL Testing Acc: {all_mll_acc.mean()*100:.4f} +/- {all_mll_acc.std()/np.sqrt(n_samples)*100:.4f}') 
        print(f'Combined Testing Acc: {all_combined_acc.mean()*100:.4f} +/- {all_combined_acc.std()/np.sqrt(n_samples)*100:.4f}') 
        print('***********************************')    
    

