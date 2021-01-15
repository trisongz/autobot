

def check_imports():
    avail = {'colab': False, 'libs': {}}
    try:
        import google.colab
        avail['colab'] = True
    except ImportError:
        avail['colab'] = False
    
    try:
        import tensorflow
        try:
            tfv = tensorflow.__version__
        except:
            tfv = tensorflow.version.VERSION
        avail['libs']['tensorflow'] = tfv
    except ImportError:
        avail['libs']['tensorflow'] = False

    try:
        import redis
        avail['libs']['redis'] = True
    except ImportError:
        avail['libs']['redis'] = False

    try:
        import transformers
        avail['libs']['transformers'] = transformers.__version__
    except ImportError:
        avail['libs']['transformers'] = False

    try:
        import tokenizers
        avail['libs']['tokenizers'] = True
    except ImportError:
        avail['libs']['tokenizers'] = False

    try:
        import tensorflow_datasets
        avail['libs']['tfds'] = True
    except ImportError:
        avail['libs']['tfds'] = False

    try:
        from google.cloud import storage
        avail['libs']['gcs'] = True
    except ImportError:
        avail['libs']['gcs'] = False

    try:
        import smart_open
        avail['libs']['smart_open'] = smart_open.__version__
    except ImportError:
        avail['libs']['smart_open'] = False

    try:
        import ray
        avail['libs']['ray'] = ray.__version__
    except ImportError:
        avail['libs']['ray'] = False

    try:
        import torch
        avail['libs']['torch'] = torch.__version__
    except ImportError:
        avail['libs']['torch'] = False

    try:
        import tqdm
        avail['libs']['tqdm'] = tqdm.__version__
    except ImportError:
        avail['libs']['tqdm'] = False
    
    try:
        import datasets
        avail['libs']['datasets'] = datasets.__version__
    except ImportError:
        avail['libs']['datasets'] = False

    try:
        import tensorflow_datasets
        avail['libs']['tensorflow_datasets'] = tensorflow_datasets.__version__
    except ImportError:
        avail['libs']['tensorflow_datasets'] = False

    try:
        import numpy as np
        avail['libs']['numpy'] = np.__version__
    except ImportError:
        avail['libs']['numpy'] = False
    
    try:
        import wandb
        avail['libs']['wandb'] = wandb.__version__
    except ImportError:
        avail['libs']['wandb'] = False
    
    try:
        import boto3
        avail['libs']['boto3'] = True
    except ImportError:
        avail['libs']['boto3'] = False

    return avail


