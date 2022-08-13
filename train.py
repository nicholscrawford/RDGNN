import argparse

from .data_utils import pc_and_sample

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

    pc_and_sample(args.train_data_dir)
       










if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description='Train for precond classification directly from images.')
    argparse.add_common_args_to_parser(parser,
                              cuda=True,
                              result_dir=True,
                              num_epochs=True,
                              batch_size=True,
                              lr=True,
                              save_freq_iters=True,
                              log_freq_iters=True,
                              print_freq_iters=True)
                              

    parser.add_argument('--train_data_dir', required=True, action='append',
                        help='Path to directory of demos from gym.')

    args = parser.parse_args()
    main(args)