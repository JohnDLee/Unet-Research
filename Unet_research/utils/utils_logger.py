import logging

logging.basicConfig(filename="logs/console.log", 
					format= '%(asctime)s:%(filename)s:%(lineno)d:%(levelname)s:%(name)s:%(message)s', ) 
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)