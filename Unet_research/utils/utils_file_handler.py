from configparser import ConfigParser
import logging
import os
import argparse

def setup_logger(filepath):
    ''' sets up a console logger '''

  
    # set up a logger
    logging.basicConfig(filename=os.path.join(filepath, 'console.log'),
                        format= '%(asctime)s:%(filename)s:%(lineno)d:%(levelname)s:%(name)s:%(message)s', ) 
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    
    return logger

def setup_argparser():
    ''' sets up a parser for reading in data '''
    parser = argparse.ArgumentParser(description='Process filepath and files.')
    parser.add_argument('filepath', help = 'filepath to save metrics (should include .ini file)')
    parser.add_argument('file', help='filename of .ini file')
    return parser
    

def read_config(filename, section):
    '''Reads a config
        
        filename: str -> e.g. database.ini
        section: str -> e.g. postgres
        
        return: dict -> dict of values stored under section
    '''
    # create config parser
    cfg = ConfigParser()
    # read file
    cfg.read(filename)
    
    # read info in section
    info = {}
    if cfg.has_section(section):
        # change params to a dict
        params = cfg.items(section)
        for param in params:
            try:
                
                if (float(param[1]) == int(param[1])): # get ints
                    info[param[0]] = int(param[1])
                continue
            except Exception as e:
                pass
            try:
                info[param[0]] = float(param[1]) # get floats
                continue
            except:
                pass
            
            if (param[1] == 'True'):
                info[param[0]] = True
            elif (param[1] == 'False'):
                info[param[0]] = False
            else:
                info[param[0]] = param[1]
                    
            
    return info




