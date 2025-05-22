import argparse
import dataset_loader
import model
import make_training_data

def main(args):
    # create dataloader
    loader = args.dataset.replace('-', '_')
    LoaderClass = getattr(dataset_loader, f'{loader}Loader')
    Dataset = LoaderClass(args)
    
    # load model
    ModelClass = getattr(model, args.model.replace('-', '_'))
    Model = ModelClass(args)

    # generate data
    MakerClass = getattr(make_training_data, f'make_{loader}_train')
    MakerClass(args, Dataset, Model)
    # import pdb; pdb.set_trace()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['MMLU', 'BBH', 'Gsm8k', 'Comqa', 'StraQA', 'MathQA', 'Reclor', 'PiQA', 'LogiQA', 'Folio', 'ARC-C'], type=str, help='Name of the dataset loader to use')
    parser.add_argument('--model', required=True, choices=['llama1-7b', 'llama1-13b', 'llama1-30b', 'llama1-65b', 'llama3-8b'], type=str, help='Name of the model to use')
    parser.add_argument('--layers', required=True, type=int, help='eg. layers=16, 0~16layers are considered in data making')
    parser.add_argument('--head', action='store_true', help='save hidden states')

    # special
    parser.add_argument('--task', default='', help='Assign specific tasks in MMLU dataset')
    args = parser.parse_args()

    main(args)
