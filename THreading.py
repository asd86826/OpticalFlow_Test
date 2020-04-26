import time 
from threading import Timer

i = 0

class RepeatedTimer(object):
    def __init__(self, interval, function, *args, **kwargs):
        self._timer     = None
        self.interval   = interval
        self.function   = function
        self.args       = args
        self.kwargs     = kwargs
        self.is_running = False
        self.start()                        #if you dont want auto start, delte that

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False

def timeTest():
    global i
    i = i+1
    print ("Hello %d!" % i)


if __name__ == "__main__":
    print("Starting...")

    rt = RepeatedTimer(0.05, timeTest) # it auto start ,so dont need rt.start()

    try:
        ST = time.time()
        time.sleep(5)
    except Exception as e:
        raise e
    finally:
        rt.stop()
        print(time.time() - ST)
