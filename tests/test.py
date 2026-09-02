def mian():
    d = dict({0:[]})
    c = d.values()
    print(c)
    d[1]= [1]
    print(c)
    print(list(d.values()))

if __name__ == "__main__":
    mian()