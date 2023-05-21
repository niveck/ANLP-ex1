import os


def prepare():
    os.environ['HF_DATASETS_CACHE'] = 'hf_cache/datasets'
    os.environ['HF_METRICS_CACHE'] = 'hf_cache/metrics'
    os.environ['HF_MODULES_CACHE'] = 'hf_cache/modules'
    os.environ['HF_DATASETS_DOWNLOADED_EVALUATE_PATH'] = 'hf_cache/datasets_downloaded_evaluate'
    os.environ['TRANSFORMERS_CACHE'] = 'transformers_cache'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_USE_CUDA_DSA'] = '1'
