import torch
import os
import argparse



def main(args: argparse.Namespace):
    rdgnn = torch.load(args.saved_model)
    pass

if __name__ == '__main__':
    

    parser = argparse.ArgumentParser(
        description='Train for dynamics model from isaacgym data which records sensor data and actions.')
                              
    parser.add_argument('--saved_model', type=str, required=True)
    
    args = parser.parse_args()
    main(args)
