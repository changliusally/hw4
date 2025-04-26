import numpy as np


### No need to modify Identity class
class Identity:
    """
    Identity activation function.
    """

    def forward(self, Z):
        """
        :param Z: Batch of data Z (N samples, C features) to apply activation function to input Z.
        :return: Output returns the computed output A (N samples, C features).
        """
        self.A = Z
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt post-activation output (a measure of how the output A affect the loss L)
        :return: Gradient of loss with respect to pre-activation input (a measure of how the input Z affect the loss L)
        """
        dAdZ = np.ones(self.A.shape, dtype="f")
        dLdZ = dLdA * dAdZ
        return dLdZ


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        
        # TODO: Implement forward pass
        # Compute the softmax in a numerically stable way
        # Apply it to the dimension specified by the `dim` parameter
        # move the dim to the last dimension to organize the rest dimensions easily
        # in case we got negative self.dim
        dim = self.dim if self.dim >= 0 else len(Z.shape) + self.dim
        Z = np.moveaxis(Z, dim, -1)

        # Numerically stable softmax
        Z_stable = Z - np.max(Z, axis=-1, keepdims=True)
        exp_Z = np.exp(Z_stable)
        self.A = exp_Z / np.sum(exp_Z, axis=-1, keepdims=True)

        self.A = np.moveaxis(self.A, -1, dim)
        return self.A

        

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # TODO: Implement backward pass
        
        # Get the shape of the input
        shape = self.A.shape
        # Find the dimension along which softmax was applied
        C = shape[self.dim]
        # Move softmax axis to last for 2D reshaping
        A_moved = np.moveaxis(self.A, self.dim, -1)
        dLdA_moved = np.moveaxis(dLdA, self.dim, -1)
        N = np.prod(A_moved.shape[:-1])


        # Reshape input to 2D (N, C)
        if len(shape) > 2:
            A_2D = A_moved.reshape(N, C)
            dLdA_2D = dLdA_moved.reshape(N, C)

            dLdZ_2D = np.zeros((N, C))  # TODO

            # common operation in HW1P1
            for i in range(N):
                a = A_2D[i]
                J = np.diag(a) - np.outer(a, a)
                dLdZ_2D[i] = dLdA_2D[i] @ J


            # Restore shapes to original
            dLdZ_moved = dLdZ_2D.reshape(A_moved.shape)
            dLdZ = np.moveaxis(dLdZ_moved, -1, self.dim)
        else:
            # only 2 dimensions
            N = shape[0]
            dLdZ = np.zeros(shape)
            for i in range(N):
                a = self.A[i]
                J = np.diag(a) - np.outer(a, a)
                dLdZ[i] = dLdA[i] @ J

        return dLdZ
    