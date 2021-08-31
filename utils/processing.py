import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def perform_PCA(df: pd.DataFrame):
    '''
    Use scikit learn's pca analysis tools to reduce dimensionality of input
    Jeff Zimmerman
    '''

    weighted_standard = StandardScaler().fit_transform(df)

    pca = PCA(n_components='mle')
    pca.fit(weighted_standard)
    # decomposed = pca.transform(weighted_standard)

    print(pca.explained_variance_ratio_)
    print(pca.components_)

    plt.plot(decomposed)
    plt.show()
    return