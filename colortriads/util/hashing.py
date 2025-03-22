import math

class NHash(object):

    def __init__(self, N, nbins=100.0):
        self.N = N
        self.nbins = nbins
        self.items = dict([])


    def get_hash(self, vec):
        res = 0
        for x in vec:
            if x < 0 or x > 1.0:
                raise ValueError('NHash takes values between 0 and 1 (got %0.3f)' % x)
            res = res << 10
            res += int(math.floor(x * self.nbins))
        return res


    def unhash(self, h):
        vec = []
        for i in range(self.N):
            h_tmp = (h >> 10)
            diff = h - (h_tmp << 10)
            vec.insert(0, diff / self.nbins)
            h = h_tmp
        return vec


    def add(self, vec):
        h = self.get_hash(vec)
        if h in self.items:
            self.items.append(vec)
        else:
            self.items[h] = [vec]
        return h


    def get_closest(self, vec):
        h = self.get_hash(vec)
        raise RuntimeError("Not implemented")
