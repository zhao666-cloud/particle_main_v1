import argparse
import sys

def get_args():
    #hyperparameters
    parser = argparse.ArgumentParser(description='Reconstruction')
    parser.add_argument('--num_particles',type=int,default=2000,help='the number of generate particles')
    parser.add_argument('--num_obj',type=int,default=1,help='the number of loading object')
    parser.add_argument('--batch_size', type=int, default=10, help='the train batch_size')
    parser.add_argument('--objects_dir',type=str,default='objects_model',help='the directory of objects')
    args = parser.parse_args(sys.argv[1:])
    return args