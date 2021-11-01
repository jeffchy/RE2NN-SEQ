from src_seq.utils import create_datetime_str, mkdir
import pickle

def save_model_and_log(logger, result, args):
    datetime_str = create_datetime_str()
    mkdir('../model_seq/')
    mkdir('../model_seq/{}'.format(args.run))

    print('Saving Args and Results')
    file_save_path = "../model_seq/{}/{}.res".format(
        args.run, datetime_str,
    )
    print('Saving Args and Results at: {}'.format(file_save_path))
    pickle.dump({
        'args': args,
        'res': result,
        'logger': logger
    }, open(file_save_path, 'wb'))