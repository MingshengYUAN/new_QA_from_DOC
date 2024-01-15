import logging

logger = logging.getLogger('root')
logger.setLevel(logging.INFO)

# 创建FileHandler对象
fh = logging.FileHandler('./log/qa_from_doc_init.log')
fh.setLevel(logging.INFO)

#
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 创建Formatter对象
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 将FileHandler对象添加到Logger对象中
logger.addHandler(fh)
logger.addHandler(console_handler)