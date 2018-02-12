import _pickle as pickle
objs = []
f = open("result.p","rb")
while 1:
    try:
        objs.append(pickle.load(f))
    except EOFError:
        break
f.close()
print(objs)
