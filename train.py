import argparse
import os
import sys

import torch
from colorama import Back, Fore, Style

from data_utils import pc_and_sample
from Dataloader import Dataloader
from RDGNN import RDGNN
from RDGNN_Config import RDGNN_Config

######################################################################################
# RDGNN Rewrite for training method. 

# The hope is to gain a better understanding of rdgnn implementation.

# Need to get input data folder,

# Manipulate into point clouds and sample if needed,

# Transform it into a graph and add one-hot encodings

# Encode graph to latent state,

# Predict graphs future latent state given action,

#######################################################################################
def main(args : argparse.Namespace) -> None :

    #Create point clouds and sample them from collected gym data.
    pc_and_sample(args.train_data_dir)

    dataloader = Dataloader(args.train_data_dir)
    rdgnn = RDGNN(RDGNN_Config())

    for epoch in range(args.epochs):
        loss = rdgnn.run_model(dataloader)
        rdgnn.update_weights(loss)

        sys.stdout.write(f"Epoch {epoch} with loss {loss}")
        if epoch % 10 == 0:
            sys.stdout.write('\n')
        else:
            sys.stdout.write('\r')

    sys.stdout.write('\n')
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    path = os.path.join(args.checkpoint_dir, "saved_model.pkl")
    torch.save(rdgnn, path)
    print(f'{Fore.GREEN}{Back.BLACK}[INFO] Saving model to: {args.checkpoint_dir}{Style.RESET_ALL}')



       










if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description='Train for dynamics model from isaacgym data which records sensor data and actions.')
                              

    parser.add_argument('--train_data_dir', required=True, action='append',
                        help='Path to directory of demos from gym.')
    parser.add_argument('--epochs', required=False, type=int, default=80)
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    args = parser.parse_args()
    main(args)
