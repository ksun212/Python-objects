obj = object()

class Ten:
    def __get__(self, obj, objtype=None):
        return "1"
class A:
    x = 5                       # Regular class attribute
    y = Ten()                   # Descriptor instance


a = A()
a.y = "1"
a.y + 1

    