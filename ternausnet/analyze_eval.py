import numpy as np
import os
import pickle
import argparse

""" Analyzes the output produced by test.py
    Computes the mean jaccard, the std and the best prediction """


def main():

    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--path_to_metrics', default="ternausnet/eval/jaccard.txt", type=str)
    arg('--num_classes', default=1, type=int)
    
    args = parser.parse_args()

    file_to_open = os.path.join(os.pardir, args.path_to_metrics)
    
    with open(file_to_open, "rb") as fp:
            metrics = pickle.load(fp)

    metrics_array = np.array(metrics)
    mean = metrics_array.mean(axis=0)
    std = metrics_array.std(axis=0)
    max_ = metrics_array.max(axis=0)

    
    if args.num_classes > 1:
        print("Jaccard neurons | mean : {mean}, std : {std}, max: {max}".format(mean=mean[0], std=std[0], max=max_[0]))
        print("Jaccard axons | mean : {mean}, std : {std}, max: {max}".format(mean=mean[1], std=std[1], max=max_[1]))
    else:
        print("Jaccard neurons | mean : {mean}, std : {std}, max: {max}".format(mean=mean, std=std, max=max_))


if __name__ == '__main__':
    main()