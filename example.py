# code copied from
# https://towardsdatascience.com/simple-transformers-introducing-the-easiest-bert-roberta-xlnet-and-xlm-library-58bf8c59b2a3
import logging
import argparse
import pandas as pd
from simpletransformers.classification import ClassificationModel
import time

def get_logger():
    # create logger
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)
    return logger

def data_prep(prefix, logger):
    """
    prepares data for experiment
    @param prefix: directory holding train and test data
    @return dictionary of train, evaluation, and test data
    """
    train_df = pd.read_csv(prefix + 'train.csv', header=None)
    logger.debug("train data sample")
    logger.debug(train_df.sample(n=5, random_state=1729))
    
    eval_df = pd.read_csv(prefix + 'test.csv', header=None)
    logger.debug("eval data sample")
    logger.debug(eval_df.sample(n=5))

    train_df[0] = (train_df[0] == 2).astype(int)
    eval_df[0] = (eval_df[0] == 2).astype(int)

    train_df = pd.DataFrame({
        'text': train_df[1].replace(r'\n', ' ', regex=True),
        'label':train_df[0]
    })

    logger.debug("train data sample after type conversion and strip newlines")
    logger.debug(train_df.sample(n=5))

    eval_df = pd.DataFrame({
        'text': eval_df[1].replace(r'\n', ' ', regex=True),
        'label':eval_df[0]
    })
    logger.debug("eval data sample after type conversion and strip newlines")
    print(eval_df.sample(n=5))
    return {'eval_df': eval_df, 'train_df': train_df}

def train_and_test(model_data, logger):
    """
    trains and tests transformer model
    @param model_data: model data for sentiment analysis
    should have train and eval dataframes
    @return: dictionary of model parameters and results
    """
    train_df = model_data['train_df']
    eval_df = model_data['eval_df']

    # Create a TransformerModel
    model = ClassificationModel('roberta', 'roberta-base')
    # Train the model
    logger.debug("start model training")
    model.train_model(train_df)
    logger.debug("complete model training")
    
    # Evaluate the model
    logger.debug("begin model evaluation")
    result, model_outputs, wrong_predictions = model.eval_model(eval_df)
    logger.debug("complete model evaluation")
    return {'model_outputs': model_outputs,            
            'result' : result}
    
# create argument parser
parser = argparse.ArgumentParser(description='transformer example')


if __name__ == "__main__":
    logger = get_logger()

    logger.debug('starting up')
    
    parser.add_argument('prefix', type=str,
                        help='directory holding train and test data')
    args = parser.parse_args()

    logger.debug("begin data preparation")
    model_data = data_prep(args.prefix, logger)
    logger.debug("complete data preparation")

    logger.debug("begin train and test")
    results = {}
    results[0] = train_and_test(model_data, logger)
    logger.debug("complete train and test")
        
    with open("results_dictionary-{}.txt".format(time.time()), "w") as results_file:
        print(results, file = results_file)
    logger.debug("all complete")
