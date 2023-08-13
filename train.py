import argparse
from machine import Machine

def main():
    parser = argparse.ArgumentParser(description='Machine learning cmd line')
    parser.add_argument('path',  action="store",  type=str)
    parser.add_argument('--save_dir', action='store', dest='save_dir', type=str)
    parser.add_argument('--arch', action='store', dest='arch', type=str, default='densenet121', help="[densenet121 | vgg13]")
    parser.add_argument('--learning_rate', action='store', dest='learning_rate', type=float, default=0.003)
    parser.add_argument('--hidden_units', action='store',dest="hidden_units", type=int, default=250)
    parser.add_argument('--epochs', action='store', dest='epochs', type=int, default=5)
    parser.add_argument('--gpu', action='store_true', dest='gpu')

    args = parser.parse_args()

    m = Machine(data_dir=args.path,
                save_dir=args.save_dir,
                epochs=args.epochs,
                arch=args.arch,
                hidden_units=args.hidden_units,
                learning_rate=args.learning_rate,
                gpu=args.gpu
                )
    print('Training...')
    m.train()
    print('Testing...')
    m.test()
    print('Saving...')
    m.save_chkpth()


if __name__ == '__main__': main() 