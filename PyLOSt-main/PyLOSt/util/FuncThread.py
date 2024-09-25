# coding=utf-8
'''
Created on Apr 5, 2018

@author: ADAPA
'''
import threading


class FuncThread(threading.Thread):
    """
    Threading class used for wrapping a function in a thread and to run it in parallel
    """
    finished = False
    retObj = None
    retVal = False

    def __init__(self, *args, **kwargs):
        self._target = kwargs.pop('target')
        self._callback = kwargs.pop('callback')
        self._callback_args = kwargs.pop('callback_args')
        self._args = args
        super(FuncThread, self).__init__(target=self.target_with_callback)
        # threading.Thread.__init__(self)

    def start(self):
        threading.Thread.start(self)
        self.target_with_callback()

    def target_with_callback(self):
        try:
            # self._target(*self._args)
            self.retObj = self._target(*self._args)
            self.retVal = True
        except Exception as e:
            print('run <- FuncThread')
            print(e)
            self.retVal = False
        finally:
            self.finished = True
            if self._callback is not None:
                if self._callback_args is None:
                    self._callback(self.retObj)
                else:
                    self._callback(self.retObj, *self._callback_args)
