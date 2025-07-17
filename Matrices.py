def VectorSum(a, b):
    """
    Returns sum of vectors a and b as a vector
    """

    if len(a) == len(b):
        c = [a[n] + b[n] for n in range(len(a))]
        return c
    else:
        raise ValueError(f'Vectors must be same length ({len(a)} and {len(b)}) are not equal.')

def VectorDifference(a, b):
    """
    Returns the difference of vectors a and b as a vector
    """

    if len(a) == len(b):
        c = [a[n] - b[n] for n in range(len(a))]
        return c
    else:
        raise ValueError(f'Vectors must be same length ({len(a)} and {len(b)}) are not equal.')


def DotProduct(a, b):
    """
    Returns the dot product of two vectors as a scalar
    """
    # print(f'dot product of {a} and {b}')
    if len(a) == len(b):
        # print([a[n] * b[n] for n in range(len(a))])
        c = sum([a[n] * b[n] for n in range(len(a))])
        return c
    else:
        raise ValueError(f'Vectors must be same length ({len(a)} and {len(b)}) are not equal.')

def MatrixSum(A, B):
    rows = len(A)
    cols = len(A[0])
    C = [[None for c in range(cols)] for r in range(rows)]
    for i in range(rows):
        for j in range(cols):
            C[i][j] = A[i][j] + B[i][j]
    return C


def MatrixMultiplyVector(W, x):
    """
    Multiplication of a matrix W by a vector x, returns a matrix
    """
    Wrows = len(W)
    Wcols = len(W[0])
    Z = [None for i in range(Wrows)]
    for n in range(Wrows):
        Z[n] = DotProduct(W[n], x)
    return Z

def VectorMultiplyMatrix(x, W):
    """
    Multiplication of a vector x by a matrix X, returns a vector
    """
    xcols = len(x)
    Wrows = len(W)
    Wcols = len(W[0])

    Z = [None for i in range(Wcols)]
    print(x, W, Z)
    for n in range(Wcols):
        w = [W[i][n] for i in range(Wrows)]
        print(x, w)
        Z[n] = DotProduct(x, w)

    return Z


def MatrixMultiply(A, B):
    Arows = len(A)
    Acols = len(A[0])
    Brows = len(B)
    Bcols = len(B[0])
    # A (rxc) x B (r x c) = C (Ar x Bc)
    C = [[None for bc in range(Bcols)] for ar in range(Arows)]

    for i in range(len(C)):
        for j in range(len(C[i])):
            C[i][j] = DotProduct(A[i], [B[r][j] for r in range(len(B))])

    return C


def Transpose(A):
    Arows = len(A)
    Acols = len(A[0])

    AT = [[None for r in range(Arows)] for c in range(Acols)]

    for i in range(Arows):
        for j in range(Acols):
            AT[j][i] = A[i][j]

    return AT


def Submatrix(A, i, j):
    m = [r[:j] + r[j+1:] for r in (A[:i] + A[i+1:])]
    return m

def Determinant(A):
    
    if len(A) == 1:
        return A[0][0]

    if len(A) == 2:
        return A[0][0] * A[1][1] - A[0][1] * A[1][0]

    determinant = 0
    for c in range(len(A)):
        determinant += ((-1)**c) * A[0][c] * Determinant(Submatrix(A, 0, c))

    return determinant

def Cofactor(A):
    Arows = len(A)
    Acols = len(A[0])
    C = [[None for c in range(Acols)] for r in range(Arows)]

    for r in range(Arows):
        for c in range(Acols):
            C[r][c] = ((-1)**(r+c)) * Determinant(Submatrix(A, r, c))

    return C


def Inverse(A):
    determinant = Determinant(A)
    adjugate = Transpose(Cofactor(A))
    # print(determinant, adjugate)
    # inverse = [[None for c in range(len(A[0]))] for r in range(len(A))]
    # for r in range(len(A)):
    #     for c in range(len(A[0])):
    #         inverse[r][c] = adjugate[r][c]/determinant
    inverse = ScalarMultiplyMatrix((1/determinant), adjugate)
    return inverse


def ScalarMultiplyMatrix(s, A):
    B = [[None for c in range(len(A[0]))] for r in range(len(A))]
    for r in range(len(A)):
        for c in range(len(A[0])):
            B[r][c] = s * A[r][c]

    return B

def CalculateWeights(X, y):
    # (XTX)-1Xty
    XT = Transpose(X)
    XTX = MatrixMultiply(XT, X)
    XTX_inverse = Inverse(XTX)
    XTY = MatrixMultiply(XT, y)
    # print(f"XT: {XT}")
    # print(f"XTX {XTX}")
    # print(f"XTX_Inverse {XTX_inverse}")
    # print(f"XTY {XTY}")

    w = MatrixMultiply(XTX_inverse, XTY)
    return w


def CalculateWeightsRegularised(X, y, g=0.01):
    # (XTX)-1Xty
    print(f"y: {y}")
    XT = Transpose(X)
    print(f"XT: {XT}")
    XTX = MatrixMultiply(XT, X)
    print(f"XTX {XTX}")
    In = IdentityMatrix(len(XTX))
    gIn = ScalarMultiplyMatrix(g, In)
    print(f"gIn {gIn}")
    XTX_gIn = MatrixSum(XTX, gIn)
    print(f"XTX_gIn {XTX_gIn}")
    XTX_gIn_inverse = Inverse(XTX_gIn)
    print(f"XTX_gIn_Inverse {XTX_gIn_inverse}")
    XTY = MatrixMultiply(XT, y)
    print(f"XTY {XTY}")

    w = MatrixMultiply(XTX_gIn_inverse, XTY)
    return w


def addConstant(A):
    Arows = len(A)
    Acols = len(A[0])

    Anew = [[None for c in range(Acols+1)] for r in range(Arows)]

    for i in range(Arows):
        Anew[i][0] = 1
        for j in range(Acols):
            Anew[i][j+1] = A[i][j]

    return Anew

def IdentityMatrix(n):
    In = [[1 if c==r else 0 for c in range(n)] for r in range(n)]

    return In


a = [1, 2, 3]
b = [1, 2, 3]
c = [1,1]
W = [[1,-1,2],
     [0,-3,1]]
x = [2, 1, 0]
A = [[0, 4, -2],[-4, -3, 0]]
B = [[0, 1], [1, -1], [2, 3]]
A = [[1,2,3],[4,5,6]]
B = [[1,2],[3,4],[5,6]]
AA = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
AAA = [[1,2,3],[4,5,6],[7,8,9]]


x = [-2,1,0]
# print(VectorSum(a, b))
# print(VectorDifference(a, b))
# print(DotProduct(a, b))
# print(MatrixMultiplyVector(AA, x))
# print(VectorMultiplyMatrix(c, W))
# print(MatrixMultiply(B, A))

# print(Transpose(B))
# print(DotProduct([0.1,0,-0.3], [-4,0.05,0.1]))
# print(MatrixMultiply(Transpose([[0.1,0,-0.3]]), [[-4,0.05,0.1]]))
# print(Determinant([[-2,-1,2],[2,1,4],[-3,3,1]]))
# print(Determinant([[1,2,3,4],[5,6,7,8],[9,1,2,3],[4,5,6,7]]))
# print(Cofactor([[9, 9, 0], [1,2,3],[4,5,6]]))
# print(Inverse([[9, 9, 0], [1,2,3],[4,5,6]]))
X = [[2,4],[10,11],[12,11],[1,1]]
y = [[4.4],[-89.9], [-118.9],[9.6]]
#print(f"Weights: {CalculateWeights(X, y)}")
Xc = addConstant(X)
#print(f"Weights with const {CalculateWeights(Xc, y)}")
#print(IdentityMatrix(4))
#print(f"Reg Weights: {CalculateWeightsRegularised(X, y)}")
#print(f"Reg Weights with const {CalculateWeightsRegularised(Xc, y)}")



A = [[2,4],[10,11],[12,11]]
B = [[4.4],[-89.9],[-118.9]]
Ac = addConstant(A)
weights = CalculateWeights(Ac, B)
weights_reg = CalculateWeightsRegularised(Ac, B, g=0.001)
print(MatrixMultiply(Ac, weights))
print(MatrixMultiply(Ac, weights_reg))

