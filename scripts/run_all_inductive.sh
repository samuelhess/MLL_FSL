# tieredImageNet mll/combined 1-shot inductive accuracy
python run_few_shot_inductive_combined_metric.py --dataset tieredImageNet --network wideres --mll_thresh 40 --n_shot 1  --compute_cov
python run_few_shot_inductive_combined_metric.py --dataset tieredImageNet --network wideres --mll_thresh 40 --n_shot 1

python run_few_shot_inductive_combined_metric.py --dataset tieredImageNet --network densenet121 --mll_thresh 40 --n_shot 1  --compute_cov
python run_few_shot_inductive_combined_metric.py --dataset tieredImageNet --network densenet121 --mll_thresh 40 --n_shot 1

python run_few_shot_inductive_combined_metric.py --dataset tieredImageNet --network resnet12 --mll_thresh 40 --n_shot 1  --compute_cov
python run_few_shot_inductive_combined_metric.py --dataset tieredImageNet --network resnet12 --mll_thresh 40 --n_shot 1

python run_few_shot_inductive_combined_metric.py --dataset tieredImageNet --network resnet18 --mll_thresh 40 --n_shot 1  --compute_cov
python run_few_shot_inductive_combined_metric.py --dataset tieredImageNet --network resnet18 --mll_thresh 40 --n_shot 1

# # ***************************************************************************************************************************************
# tieredImageNet mll/combined 5-shot inductive accuracy

python run_few_shot_inductive_combined_metric.py --dataset tieredImageNet --network wideres --mll_thresh 40 --n_shot 5  --compute_cov
python run_few_shot_inductive_combined_metric.py --dataset tieredImageNet --network wideres --mll_thresh 40 --n_shot 5

python run_few_shot_inductive_combined_metric.py --dataset tieredImageNet --network densenet121 --mll_thresh 40 --n_shot 5  --compute_cov
python run_few_shot_inductive_combined_metric.py --dataset tieredImageNet --network densenet121 --mll_thresh 40 --n_shot 5

python run_few_shot_inductive_combined_metric.py --dataset tieredImageNet --network resnet12 --mll_thresh 40 --n_shot 5  --compute_cov
python run_few_shot_inductive_combined_metric.py --dataset tieredImageNet --network resnet12 --mll_thresh 40 --n_shot 5

python run_few_shot_inductive_combined_metric.py --dataset tieredImageNet --network resnet18 --mll_thresh 40 --n_shot 5  --compute_cov
python run_few_shot_inductive_combined_metric.py --dataset tieredImageNet --network resnet18 --mll_thresh 40 --n_shot 5

# ****************************************************************************************************************************************
# ****************************************************************************************************************************************
# miniImageNet mll/combined 1-shot inductive accuracy

python run_few_shot_inductive_combined_metric.py --dataset miniImageNet --network wideres --mll_thresh 40 --n_shot 1  --compute_cov  
python run_few_shot_inductive_combined_metric.py --dataset miniImageNet --network wideres --mll_thresh 40 --n_shot 1  

python run_few_shot_inductive_combined_metric.py --dataset miniImageNet --network densenet121 --mll_thresh 40 --n_shot 1  --compute_cov 
python run_few_shot_inductive_combined_metric.py --dataset miniImageNet --network densenet121 --mll_thresh 40 --n_shot 1 

python run_few_shot_inductive_combined_metric.py --dataset miniImageNet --network resnet12 --mll_thresh 40 --n_shot 1  --compute_cov
python run_few_shot_inductive_combined_metric.py --dataset miniImageNet --network resnet12 --mll_thresh 40 --n_shot 1

python run_few_shot_inductive_combined_metric.py --dataset miniImageNet --network resnet18 --mll_thresh 40 --n_shot 1  --compute_cov
python run_few_shot_inductive_combined_metric.py --dataset miniImageNet --network resnet18 --mll_thresh 40 --n_shot 1

# ***************************************************************************************************************************************
# miniImageNet mll/combined 5-shot inductive accuracy

python run_few_shot_inductive_combined_metric.py --dataset miniImageNet --network wideres --mll_thresh 40 --n_shot 5  --compute_cov
python run_few_shot_inductive_combined_metric.py --dataset miniImageNet --network wideres --mll_thresh 40 --n_shot 5

python run_few_shot_inductive_combined_metric.py --dataset miniImageNet --network densenet121 --mll_thresh 40 --n_shot 5  --compute_cov 
python run_few_shot_inductive_combined_metric.py --dataset miniImageNet --network densenet121 --mll_thresh 40 --n_shot 5 

python run_few_shot_inductive_combined_metric.py --dataset miniImageNet --network resnet12 --mll_thresh 40 --n_shot 5  --compute_cov
python run_few_shot_inductive_combined_metric.py --dataset miniImageNet --network resnet12 --mll_thresh 40 --n_shot 5

python run_few_shot_inductive_combined_metric.py --dataset miniImageNet --network resnet18 --mll_thresh 40 --n_shot 5  --compute_cov
python run_few_shot_inductive_combined_metric.py --dataset miniImageNet --network resnet18 --mll_thresh 40 --n_shot 5