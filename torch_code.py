"""
This file and its tests are individualized for NetID vdwavhal.
"""
import numpy as np
import torch as tr

def mountain1d(x):
    """
    Input: float x
    Output: floats z, dz_dx
      z: value of -1*x**3 + 1*x**2 - 3*x - 3
      dz_dx: value of dz/dx evaluated at input
    Use torch to compute derivatives and then type-cast back to float
    """
    def f(val):
        ans = -3*val**3 + 3*val**2 + 3*val + 1
        return ans

    y = tr.tensor(x,requires_grad=True)
    z = f(y)
    z.backward()
    dz = y.grad
    z = float(z)
    dz = float(dz)
    return z,dz

def robot(t1, t2):
    """
    Input: floats t1, t2
        each joint angle, in units of radians
    Output: floats z, dz_dt1, dz_dt2
      z: value of (x - -1)**2 + (y - 2)**2, where
         x = 3*(cos(t1)*cos(t2) - sin(t1)*sin(t2)) + 4*cos(t1)
         y = 3*(sin(t1)*cos(t2) + cos(t1)*sin(t2)) + 4*sin(t1)
      dz_dt1, dz_dt2: values of dz/dt1 and dz/dt2 evaluated at input
    Use torch to compute derivatives and then type-cast back to float
    """
    def f(x,y):
        return 3*(tr.cos(x)*tr.cos(y) - tr.sin(x)*tr.sin(y)) + 2*tr.cos(x)
    def g(x,y):
        return 3*(tr.sin(x)*tr.cos(y) + tr.cos(x)*tr.sin(y)) + 2*tr.sin(x)
    def h(x,y):
        return (x-3)**2 + (y-1)**2


    m = tr.tensor(t1,requires_grad=True)
    n = tr.tensor(t2,requires_grad=True)
    z = h(f(m,n),g(m,n))
    z.backward()



    z = float(z)
    fx = float(m.grad)
    fy = float(n.grad)
    # TODO: replace with your implementation
    return z,fx,fy

def neural_network(W1, W2, W3):
    """
    Input: numpy arrays W1, W2, and W3 representing weight matrices
    Output: y, e, de_dW1, de_dW2, and de_dW3
        float y: the output of the neural network
        float e: the squared error of the neural network
        numpy array de_dWk: the gradient of e with respect to Wk, for k in [1, 2, 3]
    Use torch to compute derivatives and then type-cast back to floats and numpy arrays
    The following documentation may be helpful:
        https://pytorch.org/docs/stable/generated/torch.tanh.html
        https://pytorch.org/docs/stable/generated/torch.mv.html
        https://pytorch.org/docs/stable/tensors.html
        https://pytorch.org/docs/stable/generated/torch.Tensor.float.html#torch.Tensor.float
        https://pytorch.org/docs/stable/generated/torch.Tensor.numpy.html#torch.Tensor.numpy
        https://numpy.org/doc/stable/user/basics.types.html
    For more information, consult the instructions.
    """
    # TODO: replace with your implementation
    W1t = tr.tensor(W1,requires_grad=True)
    W2t = tr.tensor(W2,requires_grad=True)
    W3t = tr.tensor(W3,requires_grad=True)



    x1 = [1,1]
    x2 = [1,1,-1]
    x3 = [-1,1,1,1]



    x1t = tr.tensor(x1)
    x2t = tr.tensor(x2)
    x3t = tr.tensor(x3)



    tr.t(x1t)
    tr.t(x2t)
    tr.t(x3t)

    # 1.9960
    y_nw = (W1t * tr.tanh(x1t + (W2t * tr.tanh(x2t + (W3t*tr.tanh(x3t)).sum(axis=1))).sum(axis=1))).sum(axis=1)
    err = ((y_nw-3)**2).sum()
    err.backward()

    y_nw = float(y_nw)
    err = float(err)



    # [float(t) for t in dW1f]
    dW1f = W1t.grad
    dW2f = W2t.grad
    dW3f = W3t.grad

    dW1f.to(torch.float32)
    dW2f.to(torch.float32)
    dW3f.to(torch.float32)

    dW1 = dW1f.numpy()
    dW2 = dW2f.numpy()
    dW3 = dW3f.numpy()

    fW = (dW1,dW2,dW3)

    return (y_nw, err) + fW

if __name__ == "__main__":

    # start with small random weights
    W1 = np.random.randn(1,2).astype(np.float32) * 0.00

    W2 = np.random.randn(2,3).astype(np.float32) * 0.00
    
    W3 = np.random.randn(3,4).astype(np.float32) * 0.00
    
    # do several iterations of gradient descent
    for step in range(100):
        
        # evaluate loss and gradients
        y, e, dW1, dW2, dW3 = neural_network(W1, W2, W3)
        if step % 10 == 0: print("%d: error = %f" % (step, e))

        # take step
        eta = .1/(step + 1)
        W1 -= dW1 * eta
        W2 -= dW2 * eta
        W3 -= dW3 * eta

