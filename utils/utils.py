import argparse
import numpy as np
import json
import scipy.stats as stats
import plotly.graph_objects as go

def parse_args(type):
    if type == 'inductive':
        parser = argparse.ArgumentParser(description= 'few-shot inductive script')
        parser.add_argument('--dataset'     , default='miniImageNet',        help='miniImageNet/tiered')
        parser.add_argument('--network'       , default='wideres',      help='model: wideres/densenet121/resnet18/resnet12') 
        parser.add_argument('--mll_thresh'      , default=50, type=float,   help='clip threshold for mll')
        parser.add_argument('--n_shot'      , default=1, type=int,  help='number of labeled data in each class, same as n_support')
        parser.add_argument('--n_way'      , default=5, type=int,  help='number of classes')
        parser.add_argument('--n_query'      , default=15, type=int,  help='number of query samples in each class')   
        parser.add_argument('--compute_cov'   , action='store_true',  help='set to true to compute covariance')
        parser.add_argument('--n_samples'      , default=10000, type=int,  help='number of evaluation tests')    
        parser.add_argument('--n_processors'      , default=64, type=int,  help='number of processors to use')
    else: #transductive
        parser = argparse.ArgumentParser(description= 'few-shot transductive script')
        parser.add_argument('--dataset'     , default='miniImageNet',        help='miniImageNet/tiered')
        parser.add_argument('--network'       , default='wideres',      help='model: wideres/densenet121/resnet18/resnet12') 
        parser.add_argument('--mll_thresh'      , default=50, type=float,   help='clip threshold for mll')
        parser.add_argument('--iterations'      , default=10, type=int,   help='Number of MLL iterations')
        parser.add_argument('--alpha'      , default=0.5, type=float,   help='alpha parameter, iteration learning rate')
        parser.add_argument('--n_shot'      , default=1, type=int,  help='number of labeled data in each class, same as n_support')
        parser.add_argument('--n_way'      , default=5, type=int,  help='number of classes')
        parser.add_argument('--n_query'      , default=15, type=int,  help='number of query samples in each class')   
        parser.add_argument('--n_samples'      , default=10000, type=int,  help='number of evaluation tests')
        parser.add_argument('--balanced'       , action='store_true',  help='set to true to evaluate balanced performance')
        parser.add_argument('--dirichlet_alpha'      , default=2, type=float,   help='alpha parameter for dirichlet distribution sampling')


    return parser.parse_args()        



def compute_and_save_statistics(inclass_features,crossclass_features,group_statistics_file):
    cov_inclass = np.cov(np.transpose(inclass_features))
    u_inclass = np.mean(inclass_features,axis = 0)
    cov_crossclass = np.cov(np.transpose(crossclass_features))
    u_crossclass = np.mean(crossclass_features,axis = 0)

    stats = {}
    stats["cov_inclass"] = cov_inclass.tolist()
    stats["u_inclass"] = u_inclass.tolist()
    stats["cov_crossclass"] = cov_crossclass.tolist()
    stats["u_crossclass"] = u_crossclass.tolist()
    with open(group_statistics_file, 'w') as f:
        json.dump(stats, f)

def load_statistics(group_statistics_file):
    with open(group_statistics_file, 'r') as f:
        stats = json.load(f)

    cov_inclass = np.array(stats["cov_inclass"])
    u_inclass = np.array(stats["u_inclass"])
    cov_crossclass = np.array(stats["cov_crossclass"])
    u_crossclass = np.array(stats["u_crossclass"])
    return [cov_inclass, u_inclass, cov_crossclass, u_crossclass]

def get_histogram_and_exp_fit(data,x,class_index,feature_index):
    class_data =np.array(np.array(data[class_index]))
    class_data = class_data/np.linalg.norm(class_data,ord=2,axis=1,keepdims=1)
    X = class_data[:,feature_index]
    hist_values, edges = np.histogram(X, bins=x,density=True)
    x = edges[:-1] + np.diff(edges)/2
    P = stats.expon.fit(X)
    lam = 1/P[1]
    pdf_fitted = lam*np.exp(-lam*x)
    return [x, hist_values, pdf_fitted]

def get_histogram_and_g_fit(X,x):
    hist_values, edges = np.histogram(X, bins=x,density=True)
    x = edges[:-1] + np.diff(edges)/2
    P = stats.norm.fit(X)
    z = (x-P[0])/P[1]
    pdf_fitted = np.exp(-0.5*(z**2))/(P[1]*np.sqrt(2*np.pi))
    return [x, hist_values, pdf_fitted]

def custom_fig_helper_function(x_a,x_b,x_c,hist_a,hist_b,hist_c,fit_a,fit_b,fit_c,x_end,title_str):
    color_a = 'rgb(169, 209, 142)'
    color_b = 'rgb(66, 119, 238)'
    color_c = 'rgb(180, 132, 190)'
    fig = go.Figure()

    fig.add_trace(go.Bar(x=x_a, y=hist_a, name='class a distribution of feature', marker_color=color_a))
    fig.add_trace(go.Bar(x=x_b, y=hist_b, name='class b distribution of feature', marker_color=color_b))
    fig.add_trace(go.Bar(x=x_c, y=hist_c, name='class c distribution of feature', marker_color=color_c))

    fig.add_trace(go.Scatter(x=x_a,y=fit_a, name=f'class a exp fit', marker_color=color_a,line=dict(width=4)))
    fig.add_trace(go.Scatter(x=x_b,y=fit_b, name=f'class b exp fit', marker_color=color_b,line=dict(width=4)))
    fig.add_trace(go.Scatter(x=x_c,y=fit_c, name=f'class c exp fit', marker_color=color_c,line=dict(width=4)))


    fig.update_layout(
        title=title_str,
        title_x=0.5,
        title_y=0.95,
        font_size=15,
        xaxis_tickfont_size=18,
        xaxis = dict(
            title='Feature Value',
            titlefont_size=20,
            tickfont_size=18,
            range=[0,x_end],
        ),
        yaxis=dict(
            title='Probabiltiy Density',
            titlefont_size=20,
            tickfont_size=18,
        ),
        legend=dict(
            x=0.45,
            y=1.0,
            font_size=20,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        margin = dict(
            l= 60,
            r= 30,
            b= 0,
            t= 60,
            pad= 4
        ),
        barmode='group',
        bargap=0.15, # gap between bars of adjacent location coordinates.
        bargroupgap=0.1 # gap between bars of the same location coordinate.

    )
    return fig


def custom_fig_helper_function_scores(x_a,x_b,hist_a,hist_b,fit_a,fit_b,x_range,title_str):

    color_a = 'rgb(169, 209, 142)'
    color_b = 'rgb(66, 119, 238)'
    color_c = 'rgb(180, 132, 190)'
    fig = go.Figure()

    # fig.add_trace(go.Histogram(x=X, histnorm='probability density'))
    fig.add_trace(go.Bar(x=x_a, y=hist_a, name='intra-class distribution', marker_color=color_a))

    # fig.add_trace(go.Histogram(x=X, histnorm='probability density'))
    fig.add_trace(go.Bar(x=x_b, y=hist_b, name='cross-class distribution', marker_color=color_b))

    fig.add_trace(go.Scatter(x=x_a,y=fit_a, name=f'intra Gaussian fit', marker_color=color_a,line=dict(width=4)))

    fig.add_trace(go.Scatter(x=x_b,y=fit_b, name=f'cross Gaussian fit', marker_color=color_b,line=dict(width=4)))

    fig.update_layout(
        title=title_str,
        title_x=0.5,
        title_y=0.95,
        font_size=15,
        xaxis_tickfont_size=18,
        xaxis = dict(
            title='Score Value',
            titlefont_size=20,
            tickfont_size=18,
            range=x_range,
        ),
        yaxis=dict(
            title='Probabiltiy Density',
            titlefont_size=20,
            tickfont_size=18,
        ),
        legend=dict(
            x=0.00,
            y=1.0,
            font_size=20,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        margin = dict(
            l= 60,
            r= 30,
            b= 0,
            t= 60,
            pad= 4
        ),
        barmode='group',
        bargap=0.15, # gap between bars of adjacent location coordinates.
        bargroupgap=0.1 # gap between bars of the same location coordinate.

    )
    return fig