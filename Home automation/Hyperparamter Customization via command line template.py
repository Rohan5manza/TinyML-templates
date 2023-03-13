# One way to provide users with ability to customize hyperparameters would be to use
#command line arguments. Allow users to specify learning rate, batch size, epoch numbers

#same script can be used for all model templates

import argparse

parser=argparse.ArgumentParser(description='Train a TinyML model')

parser.add_argument('--learning_rate',type=float,default=0.001,help='Learning rate')
parser.add_argument('--batch_size',type=int,default=32,help='batch size')
parser.add_argument('--epochs',type=int,default=100,help='Number of epochs')

args=parser.parse_args()

learning_rate=args.learning_rate
batch_size=args.batch_size
epochs=args.epoch

model= TinyMLModel(learning_rate=learning_rate,batch_size=batch_size,epochs=epochs)

# here, replace TinyMLModel with name of any other ml model from list of templates
# This only works when the model has hyperparamters with same name as the command-line arguments

#Users can then run the script with custom hyperparameters as :
# python train.py --learning_rate 0.01 --batch_size 64 --epochs 200
