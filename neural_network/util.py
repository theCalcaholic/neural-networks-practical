
class Logger(object):
    DEBUG = False

    def __init__(self):
        raise Exception("This class is abstract. It cannot be initialized!")

    @classmethod
    def log(cls, s):
        print(s)

    @classmethod
    def debug(cls, s):
        if Logger.DEBUG:
            print(s)
