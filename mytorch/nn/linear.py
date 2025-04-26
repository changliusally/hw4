import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        """
        Initialize the weights and biases with zeros
        W shape: (out_features, in_features)
        b shape: (out_features,)  # Changed from (out_features, 1) to match PyTorch
        """
        # DO NOT MODIFY
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros(out_features)


    def init_weights(self, W, b):
        """
        Initialize the weights and biases with the given values.
        """
        # DO NOT MODIFY
        self.W = W
        self.b = b

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (*, in_features)
        :return: Output Z with shape (*, out_features)
        
        Handles arbitrary batch dimensions like PyTorch
        """
        # TODO: Implement forward pass
        
        # Store input for backward pass
        self.A = A
        self.input_shape    = A.shape # save original shape(*, in_features)
        self.batch_size = np.prod(A.shape[:-1])
        self.in_features = A.shape[-1]

        # Flatten A to (batch_size, in_features)
        A_flat = A.reshape(self.batch_size, self.in_features)
        self.A_flat = A_flat  # Store for backward pass

        # Affine transformation
        Z = A_flat @ self.W.T + self.b.T  # (batch_size, out_features)
        # Unflatten Z to original shape with out_features as last dim
        out_shape = (*self.input_shape[:-1], self.W.shape[0])
        
        return Z.reshape(out_shape)
        

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (*, out_features)
        :return: Gradient of loss wrt input A (*, in_features)
        """
        # TODO: Implement backward pass

        # Compute gradients (refer to the equations in the writeup)
        out_features = self.W.shape[0]
        dLdZ_flat = dLdZ.reshape(self.batch_size, out_features)
        dLdA_flat = dLdZ_flat @ self.W  # shape: (batch_size, in_features), dLdA_flat
        
        self.dLdW = dLdZ_flat.T @ self.A_flat  # shape: (out_features, in_features)
        self.dLdb = np.sum(dLdZ_flat, axis=0)  # shape: (out_features, )
        self.dLdA = dLdA_flat.reshape(self.input_shape)
        
        # Return gradient of loss wrt input
        return self.dLdA
