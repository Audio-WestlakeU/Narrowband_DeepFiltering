import keras.backend as K

def complex_linear_filtering(x): 
    n = int(x.shape[1]/4)
    a = K.sum(x[:,0:2*n:2]*x[:,2*n:4*n:2]-x[:,1:2*n:2]*x[:,2*n+1:4*n:2],axis=1) 
    b = K.sum(x[:,0:2*n:2]*x[:,2*n+1:4*n:2]+x[:,1:2*n:2]*x[:,2*n:4*n:2],axis=1) 
    return K.concatenate([K.reshape(a,[-1,1]),K.reshape(b,[-1,1])],1)


