from copy import deepcopy

def print_and_log_results(logger, results, epoch, mode):
    """
    :param logger: the info logger
    :param results: results dict contains token-level and entity-level results
    :param epoch: num epoch, int or 'init'
    :param mode: 'TEST', 'TRAIN', 'DEV'
    :return:
    """

    assert mode in ['TRAIN', 'DEV', 'TEST', 'DEV_RE', 'DEV_NO_RE']

    acc_test, p_test, r_test, f_test = results['token-level']
    info = 'TOKEN | {} EPOCH {} |  ACC: {}, P: {}, R:{}, F1: {}'.format(mode, epoch, acc_test, p_test, r_test, f_test)
    print(info)
    logger.add(info)
    acc_test_ner, p_test_ner, r_test_ner, f_test_ner, class_res = results['entity-level']
    info = 'ENTITY | {} EPOCH {} |  ACC: {}, P: {}, R:{}, F1: {}'.format(mode, epoch, acc_test_ner, p_test_ner, r_test_ner,
                                                                f_test_ner)
    print(info)
    logger.add(info)

    info = str(class_res)
    print(info)
    logger.add(info)


class Best_Model_Recorder():
    def __init__(self, selector='f', level='token-level', init_results_train=None, init_results_dev=None, init_results_test=None, save_model=False):

        selector_pool = ['p', 'r', 'f']
        assert selector in selector_pool
        assert level in ['token-level', 'entity-level']

        self.best_dev_results = init_results_dev
        self.best_dev_train_results = init_results_train
        self.best_dev_test_results = init_results_test
        self.selector = selector_pool.index(selector) + 1
        self.level = level
        self.best_selector = self.best_dev_results[self.level][self.selector]
        self.best_model_state_dict = None
        self.save_model = save_model

    def update_and_record(self, results_train, results_dev, results_test, model_state_dict,):
        temp_selector_value = results_dev[self.level][self.selector]
        if temp_selector_value > self.best_selector:
            self.best_selector = temp_selector_value
            self.best_dev_results = results_dev
            self.best_dev_test_results = results_test
            self.best_dev_train_results = results_train

            if self.save_model:
                self.best_model_state_dict = deepcopy(model_state_dict)