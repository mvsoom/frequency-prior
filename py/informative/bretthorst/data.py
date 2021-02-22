import numpy as np
import pandas

full_data = pandas.read_csv("../../../data/pb.csv")
    
male = full_data['Type'] == 'm'
data = full_data[male]

x = np.array(data['F1'])
x0 = np.amin(x)
y = np.array(data['F2'])
z = np.array(data['F3'])

u1 = np.log(x/x0)
u2 = np.log(y/x)
u3 = np.log(z/y)

def dump(file, u):
    index = 1 + np.arange(len(u))
    cols = np.vstack([index, u]).T
    np.savetxt(file, cols, ["%i", "%.16f"], " ")

if __name__ == '__main__':
    dump("u1.ascii", u1)
    dump("u2.ascii", u2)
    dump("u3.ascii", u3)