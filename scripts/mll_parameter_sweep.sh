# Run this bash script to sweep across mll clipping parameter. choose the parameter with highest validation accuracy

python run_few_shot_inductive_combined_metric.py --dataset miniImageNet --network densenet121 --mll_thresh 10 --n_shot 1  --compute_cov

python run_few_shot_inductive_combined_metric.py --dataset miniImageNet --network densenet121 --mll_thresh 20 --n_shot 1  --compute_cov

python run_few_shot_inductive_combined_metric.py --dataset tieredImageNet --network densenet121 --mll_thresh 30 --n_shot 1  --compute_cov

python run_few_shot_inductive_combined_metric.py --dataset tieredImageNet --network densenet121 --mll_thresh 40 --n_shot 1  --compute_cov

python run_few_shot_inductive_combined_metric.py --dataset tieredImageNet --network densenet121 --mll_thresh 50 --n_shot 1  --compute_cov

python run_few_shot_inductive_combined_metric.py --dataset tieredImageNet --network densenet121 --mll_thresh 60 --n_shot 1  --compute_cov

python run_few_shot_inductive_combined_metric.py --dataset tieredImageNet --network densenet121 --mll_thresh 70 --n_shot 1  --compute_cov

python run_few_shot_inductive_combined_metric.py --dataset miniImageNet --network densenet121 --mll_thresh 80 --n_shot 1  --compute_cov

python run_few_shot_inductive_combined_metric.py --dataset miniImageNet --network densenet121 --mll_thresh 90 --n_shot 1  --compute_cov

python run_few_shot_inductive_combined_metric.py --dataset miniImageNet --network densenet121 --mll_thresh 100 --n_shot 1  --compute_cov
