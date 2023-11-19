import argparse
import logging

import cascade_classifier


def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=log_file)

    # Create a handler for displaying logs on stdout
    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(logging.DEBUG)  # Set the desired logging level
    stdout_handler.setFormatter(
        logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    # Add the handlers to the root logger
    logging.getLogger('').addHandler(stdout_handler)


if __name__ == '__main__':
    numPos = 1000
    numNeg = 1000
    numStages = 50
    acceptanceRatioBreakValue = 0.000001

    # Print the usage instructions
    parser = argparse.ArgumentParser(description='Train a cascade classifier')
    parser.add_argument('-numPos',
                        type=int,
                        default=numPos,
                        help='Number of positive samples')
    parser.add_argument('-numNeg',
                        type=int,
                        default=numNeg,
                        help='Number of negative samples')
    parser.add_argument('-numStages',
                        type=int,
                        default=numStages,
                        help='Number of stages')
    parser.add_argument('-maxWeakCount',
                        type=int,
                        default=200,
                        help='Max no of weak classifiers per stage')

    parser.add_argument('-acceptanceRatioBreakValue',
                        type=float,
                        default=acceptanceRatioBreakValue,
                        help='Acceptance ratio break value')
    parser.add_argument('-model',
                        type=str,
                        required=True,
                        help='model directory to save the cascade classifier')
    parser.add_argument('-data_pos',
                        type=str,
                        required=True,
                        help='Positive samples directory')
    parser.add_argument('-data_neg',
                        type=str,
                        required=True,
                        help='Negative samples directory')
    parser.add_argument('-results_dir',
                        type=str,
                        required=True,
                        help='Results directory')

    parser.add_argument('-W',
                        type=int,
                        default=24,
                        help='Width of the samples')
    parser.add_argument('-H',
                        type=int,
                        default=24,
                        help='Height of the samples')

    parser.add_argument('-minHitRate',
                        type=float,
                        default=0.995,
                        help='Minimum hit rate')
    parser.add_argument('-maxFalseAlarmRate',
                        type=float,
                        default=0.6,
                        help='Maximum false alarm rate')

    parser.add_argument('-log_file',
                        type=str,
                        default='traincascade.log',
                        help='Log file name (default: traincascade.log)')

    parser.add_argument('--reUsePreviousTraining',
                        action='store_true',
                        default=False,
                        help='Restart previous training')

    args = parser.parse_args()
    args.train = True

    # Setup logging
    setup_logging(args.log_file)

    cascade_clf = cascade_classifier.CascadeClassifier(args)
    cascade_clf.train()