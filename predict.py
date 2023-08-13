import argparse
from machine import Machine

def main():
    parser = argparse.ArgumentParser(description='Machine learning cmd line')
    parser.add_argument('img_path',  action="store",  type=str)
    parser.add_argument('chkpt_path',  action="store",  type=str)
    parser.add_argument('--top_k', action='store', dest='top_k', type=int, default=5)
    parser.add_argument('--category_names', action='store', dest='category_names', type=str)
    parser.add_argument('--gpu', action='store_true', dest='gpu')

    args = parser.parse_args()

    m = Machine()
    m.load_checkpoint(args.chkpt_path)
    print("Predicting...")
    p,c = m.predict(args.img_path, topk=args.top_k, gpu=args.gpu)
    print(p,c)


    cat_to_name = m.cat_to_name
    if args.category_names:
        cat_to_name = args.category_names

    labels = []
    for idx in c:
        labels.append(cat_to_name[idx])

    print(labels)


if __name__ == '__main__': main() 