from torch import nn


class StatefulComponent(nn.Module):
    def reset(self):
        raise NotImplementedError


class Controller(StatefulComponent):
    def __init__(self):
        # TODO: NOT IMPLEMENTED YET
        self.embedding = nn.Embedding()
        self.cell = nn.LSTMCell()
        pass

    def forward(self, x):
        # TODO: NOT IMPLEMENTED YET
        pass

    def reset(self):
        # TODO: NOT IMPLEMENTED YET
        pass


class Head(StatefulComponent):
    def __init__(self):
        # TODO: NOT IMPLEMENTED YET
        pass

    def forward(self, embedded, m):
        raise NotImplementedError

    def reset(self):
        # TODO: NOT IMPLEMENTED YET
        pass

    def _move(self, m, k, b, g, s, r):
        w = self._find_by_content_addressing(m, k)
        w = self._interpolate_with_previous_head(w, g)
        w = self._move_by_location_addressing(w, s)
        w = self.w = self._sharpen(w, r)
        return w

    def _interpret(self, embedded):
        raise NotImplementedError

    def _interpolate_with_previous_head(self, w, g):
        # TODO: NOT IMPLEMENTED YET
        pass

    def _find_by_content_addressing(self, m, k):
        # TODO: NOT IMPLEMENTED YET
        pass

    def _move_by_location_addressing(self, w, s):
        # TODO: NOT IMPLEMENTED YET
        pass

    def _sharpen(self, w, r):
        # TODO: NOT IMPLEMENTED YET
        pass


class ReadHead(Head):
    def forward(self, embedded, m):
        k, b, g, s, r = self._interpret(embedded)
        return self._move(m, k, b, g, s, r)

    def _interpret(self, embedded):
        # TODO: NOT IMPLEMENTED YET
        pass


class WriteHead(Head):
    def forward(self, embedded, m):
        k, b, g, s, r, e, a = self._interpret(embedded)
        return self._move(m, k, b, g, s, r), e, a

    def _interpret(self, embedded):
        # TODO: NOT IMPLEMENTED YET
        pass


class Memory(StatefulComponent):
    def read(self, w):
        # TODO: NOT IMPLEMENTED YET
        pass

    def write(self, w, e, a):
        # TODO: NOT IMPLEMENTED YET
        pass

    def reset(self):
        # TODO: NOT IMPLEMENTED YET
        pass


class NTM(StatefulComponent):
    def __init__(self):
        # TODO: NOT IMPLEMENTED YET
        super().__init__()
        self.controller = Controller()
        self.read_head = ReadHead()
        self.write_head = WriteHead()
        self.memory = Memory()
        self.linear = nn.Linear()

    def forward(self, x=None, return_read_memory=False):
        embedded = self.controller(x)
        self.memory.write(*self.write_head(embedded, self.memory))
        r = self.memory.read(*self.read_head(embedded, self.memory))
        return r if return_read_memory else self.linear(r)

    def reset(self):
        self.controller.reset()
        self.read_head.reset()
        self.write_head.reset()
        self.memory.reset()
