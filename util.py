import torch
import os
import argparse
import tarfile

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epochs', type=int, default=1)
    
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-layers', type=int, default=3)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--sequence-length', type=int, default=64)
    parser.add_argument('--lstm-size', type=int, default=256)
    parser.add_argument('--embedding-dim', type=int, default=256)

    #args = parser.parse_args(args=[])  ##for colab
    args = parser.parse_args()
    return args

def get_device():
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
    return DEVICE

def set_env(root_path='.', kind='ml'):
    # for train
    if not 'SM_CHANNEL_TRAIN' in os.environ :
        os.environ['SM_CHANNEL_TRAIN'] = '%s/data-%s/' % (root_path, kind)
    if not 'SM_MODEL_DIR' in os.environ:
        os.environ['SM_MODEL_DIR'] = '%s/output/' % root_path

    # for inference
    if not 'SM_CHANNEL_EVAL' in os.environ :
        os.environ['SM_CHANNEL_EVAL'] = '%s/data-%s/' % (root_path, kind)
    if not 'SM_CHANNEL_MODEL' in os.environ :
        os.environ['SM_CHANNEL_MODEL'] = '%s/output/' % root_path
    if not 'SM_OUTPUT_DATA_DIR' in os.environ :
        os.environ['SM_OUTPUT_DATA_DIR'] = '%s/output/' % root_path

    args = get_args()

    return args

def save_model(model, model_dir):
    path = os.path.join(model_dir, 'model.pth')
    torch.save(model.state_dict(), path)
    #torch.save(model.state_dict(), model_dir)

def load_model(model, model_dir):
    tarpath = os.path.join(model_dir, 'model.tar.gz')
    if os.path.exists(tarpath):
        tar = tarfile.open(tarpath, 'r:gz')
        tar.extractall(path=model_dir)
    model_path = os.path.join(model_dir, 'model.pth')
    model.load_state_dict(torch.load(model_path))
    return model
