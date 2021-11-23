def mean_binarization(x):
	"""
	Discretiza un vector x y discretiza en base a su media.
	"""
	import numpy as np
	mu = x.mean()
	for i in range(len(x)):
		x[i] = 1 if x[i] <= mu else -1
	return x

def mean_var_discretization(x, alpha=0.5):
    """
    Discretiza un vector x en base a su media y varianza y un parametro alpha
    """
	import numpy as np
	mu = x.mean()
	var = x.var()
	for i in range(len(x)):
		if mu+alpha*var < x[i]:
			x[i] = 1
		elif x[i] < mu+alpha*var:
			x[i] = -1
		else:
			x[i] = 0
	return x

def get_discrete_mutual_information(x, y):
    """
    Obtiene la informacion mutua entre dos vectores que representan distribuciones de datos de dos variables
    """
    from scipy.stats import entropy
    import pandas as pd
    contingency_table = pd.DataFrame({'col1':x, 'col2':y}).groupby(by=['col1', 'col2'])['col1'].count().unstack().fillna(0).values
    joint_p = contingency_table/contingency_table.sum()
    p1 = (x.value_counts()/x.size)
    p2 = (y.value_counts()/y.size)
    return entropy(p1)+entropy(p2)-entropy(joint_p.flatten())

def arg_max_phi(df, X, S, y, mu_matrix, mu_table):
    """
    Obtiene la variable (columna) x que maximiza
    """"
    m = len(S)
    max_phi = -np.inf
    arg_max_x = None
    for xj in X-S:
        i = mu_table[xj]
        j = mu_table[y.name]
        if mu_matrix[i,j] == -1:
            I = get_discrete_mutual_information(df[xj], y)
            mu_matrix[i,j] = I
        else:
            I = mu_matrix[i,j]

        if m-1 <= 0:
            phi = I
        else:
            R = []
            for xi in S:
                j = mu_table[xi]
                if mu_matrix[i,j] == -1:
                    #print("computing mutual information between {} and {}".format(xj, xi))
                    mu_matrix[i,j] = get_discrete_mutual_information(df[xj], df[xi])
                    R.append(mu_matrix[i,j])
                else:
                    #print("using precomputed value between {} and {}".format(xj, xi))
                    R.append(mu_matrix[i,j])
            R = np.array(R).sum()
            phi = I-(1/(m-1))*R

        if phi >= max_phi:
            arg_max_x = xj
            max_phi = phi
    return arg_max_x, mu_matrix

def get_mrmr_columns(X, y, m, mu_matrix=None):
    from IPython.display import clear_output
    mu_table = {col:i for i, col in enumerate(X.columns.tolist()+[y.name])}
    XminusS = set(X.columns.tolist())
    target = y.name
    S = set()
    for i in range(m):
        xj, mu_matrix = arg_max_phi(X ,XminusS, S, y, mu_matrix, mu_table)
        S.add(xj)
        clear_output()
        print('Size of S:{}'.format(len(S)))
    return S, mu_matrix
