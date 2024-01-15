class ShareArgs():

	# args就是整个项目经常使用到的默认参数，经常不需要进行变动，所以就写在这里。然后通过update的更新写到启动文件的字典参数里面
    args = {
        "port": 3010,
        "config_path": './conf/config.ini',
        "log_path": './log/qa_from_doc.log',
    }

    def get_args():  # 获取参数字典
        return ShareArgs.args

    def set_args(args):  # 一次性更新修改所有参数字典的值
        ShareArgs.args = args

    def set_args_value(key, value):  # 根据索引更新参数字典的值
        ShareArgs.args[key] = value

    def get_args_value(key, default_value=None):  # 获取指定索引的默认参数的值
        return ShareArgs.args.get(key, default_value)

    def contain_key(key):  # 判断索引是否在参数字典里面
        return key in ShareArgs.args.keys()

    def update(args):  # 用于更新字典中的键/值对，可以修改存在的键对应的值，也可以添加新的键/值对到字典中
        ShareArgs.args.update(args)
