import torch
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epochs', type=int, default=10)
    
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

def set_env(root_path='.'):
    # for train
    if not 'SM_CHANNEL_TRAIN' in os.environ :
        os.environ['SM_CHANNEL_TRAIN'] = '%s/data/train_data.txt' % root_path
    if not 'SM_MODEL_DIR' in os.environ:
        os.environ['SM_MODEL_DIR'] = '%s/output/model.pth' % root_path

    # for inference
    if not 'SM_CHANNEL_EVAL' in os.environ :
        os.environ['SM_CHANNEL_EVAL'] = '%s/data/train_data.txt' % root_path
    if not 'SM_CHANNEL_MODEL' in os.environ :
        os.environ['SM_CHANNEL_MODEL'] = '%s/output/model.pth' % root_path
    if not 'SM_OUTPUT_DATA_DIR' in os.environ :
        os.environ['SM_OUTPUT_DATA_DIR'] = '%s/output/result.txt' % root_path

    args = get_args()

    return args

def save_model(model, model_dir):
    #path = os.path.join(model_dir, 'model.pth')
    #torch.save(model.state_dict(), path)
    torch.save(model.state_dict(), model_dir)

def load_model(model, model_dir):
    #tarpath = os.path.join(saved_model, 'model.tar.gz')
    #tar = tarfile.open(tarpath, 'r:gz')
    #tar.extractall(path=saved_model)

    #model_path = os.path.join(model_dir, 'model.pth')
    model.load_state_dict(torch.load(model_dir))

    return model