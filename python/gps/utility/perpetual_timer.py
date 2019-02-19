from threading import Timer

class PerpetualTimer():

    def __init__(self, t, hFunction):
        self.t = t
        self.hFunction = hFunction
        self.thread = Timer(self.t, self.handle_function)
        print("handle_function: ", self.handle_function())

    def handle_function(self):
        self.hFunction()
        self.thread = Timer(self.t, self.handle_function)
        self.thread.start()

    def start(self):
        if not self.thread.is_alive():
            self.thread.start()

    def cancel(self):
        self.thread.cancel()