class Mapping:
    def __init__(self, iterable):
        self.items_list = []
        self.__update(iterable)

    def update(self, iterable):
        for item in iterable:
            self.items_list.append(item)

    __update = update   # private copy of original update() method

class MappingSubclass(Mapping):
    def update(self, keys, values):
        # provides new signature for update()
        # but does not break __init__()
        for item in zip(keys, values):
            self.items_list.append(item)

x = Mapping('TTH')
print(x.items_list)
x.update('123')
print(x.items_list)

s = MappingSubclass('ABC')
print(s.items_list)
s.update('THE','DEF')
print(s.items_list)