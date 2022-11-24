
# tieredImageNet mll/ transductive 1-shot 

python run_few_shot_transductive_mll_metric.py --dataset tieredImageNet --network wideres --mll_thresh 40 --iterations 10 --alpha 0.5 --n_shot 1 --n_way 5 --n_query 15 --n_samples 10000
python run_few_shot_transductive_mll_metric.py --dataset tieredImageNet --network densenet121 --mll_thresh 40 --iterations 10 --alpha 0.5 --n_shot 1 --n_way 5 --n_query 15 --n_samples 10000
python run_few_shot_transductive_mll_metric.py --dataset tieredImageNet --network resnet12 --mll_thresh 40 --iterations 10 --alpha 0.5 --n_shot 1 --n_way 5 --n_query 15 --n_samples 10000
python run_few_shot_transductive_mll_metric.py --dataset tieredImageNet --network resnet18 --mll_thresh 40 --iterations 10 --alpha 0.5 --n_shot 1 --n_way 5 --n_query 15 --n_samples 10000

# ****************************************************************************************************************************************************************************************
# tieredImageNet mll/ transductive 5-shot 
python run_few_shot_transductive_mll_metric.py --dataset tieredImageNet --network wideres --mll_thresh 40 --iterations 10 --alpha 0.5 --n_shot 5 --n_way 5 --n_query 15 --n_samples 10000
python run_few_shot_transductive_mll_metric.py --dataset tieredImageNet --network densenet121 --mll_thresh 40 --iterations 10 --alpha 0.5 --n_shot 5 --n_way 5 --n_query 15 --n_samples 10000
python run_few_shot_transductive_mll_metric.py --dataset tieredImageNet --network resnet12 --mll_thresh 40 --iterations 10 --alpha 0.5 --n_shot 5 --n_way 5 --n_query 15 --n_samples 10000
python run_few_shot_transductive_mll_metric.py --dataset tieredImageNet --network resnet18 --mll_thresh 40 --iterations 10 --alpha 0.5 --n_shot 5 --n_way 5 --n_query 15 --n_samples 10000

# ****************************************************************************************************************************************************************************************
# ****************************************************************************************************************************************************************************************
# miniImageNet mll transductive 1-shot

python run_few_shot_transductive_mll_metric.py --dataset miniImageNet --network wideres --mll_thresh 40 --iterations 10 --alpha 0.5 --n_shot 1 --n_way 5 --n_query 15 --n_samples 10000
python run_few_shot_transductive_mll_metric.py --dataset miniImageNet --network densenet121 --mll_thresh 40 --iterations 10 --alpha 0.5 --n_shot 1 --n_way 5 --n_query 15 --n_samples 10000
python run_few_shot_transductive_mll_metric.py --dataset miniImageNet --network resnet12 --mll_thresh 40 --iterations 10 --alpha 0.5 --n_shot 1 --n_way 5 --n_query 15 --n_samples 10000
python run_few_shot_transductive_mll_metric.py --dataset miniImageNet --network resnet18 --mll_thresh 40 --iterations 10 --alpha 0.5 --n_shot 1 --n_way 5 --n_query 15 --n_samples 10000

# ****************************************************************************************************************************************************************************************
# miniImageNet mll transductive 5-shot

python run_few_shot_transductive_mll_metric.py --dataset miniImageNet --network wideres --mll_thresh 40 --iterations 10 --alpha 0.5 --n_shot 5 --n_way 5 --n_query 15 --n_samples 10000
python run_few_shot_transductive_mll_metric.py --dataset miniImageNet --network densenet121 --mll_thresh 40 --iterations 10 --alpha 0.5 --n_shot 5 --n_way 5 --n_query 15 --n_samples 10000
python run_few_shot_transductive_mll_metric.py --dataset miniImageNet --network resnet12 --mll_thresh 40 --iterations 10 --alpha 0.5 --n_shot 5 --n_way 5 --n_query 15 --n_samples 10000
python run_few_shot_transductive_mll_metric.py --dataset miniImageNet --network resnet18 --mll_thresh 40 --iterations 10 --alpha 0.5 --n_shot 5 --n_way 5 --n_query 15 --n_samples 10000