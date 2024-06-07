class ShareArgs():

    args = {
        "port": 3010,
        "config_path": './conf/config.ini',
        "log_path": './log/qa_from_doc.log',
    }

    def get_args():  # get para dict
        return ShareArgs.args

    def set_args(args):  # update all para dict
        ShareArgs.args = args

    def set_args_value(key, value):  # update para dict accroding to the key
        ShareArgs.args[key] = value

    def get_args_value(key, default_value=None):  # get para dict accroding to the key
        return ShareArgs.args.get(key, default_value)

    def contain_key(key):  # determine if the key exists
        return key in ShareArgs.args.keys()

    def update(args):  # update key
        ShareArgs.args.update(args)

class LLMArgs():

    args = {
        "stream": True,
        "temperature": 0.0,
        "max_tokens": 2048
    }

    def get_args():  # get para dict
        return LLMArgs.args

    def set_args(args):  # update all para dict
        LLMArgs.args = args

    def set_args_value(key, value):  # update para dict accroding to the key
        LLMArgs.args[key] = value

    def get_args_value(key, default_value=None):  # get para dict accroding to the key
        return LLMArgs.args.get(key, default_value)

    def contain_key(key):  # determine if the key exists
        return key in LLMArgs.args.keys()

    def update(args):  # update key
        LLMArgs.args.update(args)



