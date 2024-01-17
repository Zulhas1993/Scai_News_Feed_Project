import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Example documents
document1 = "Import Numpy as np is a library that provides a set of high level functions and features for performing data analysis and manipulation. It enables you to process, analyze, manipulate, and visualize large amounts of data quickly and efficiently."
document2 = "Import Numpy as np is an incredibly powerful and versatile numerical computing library for Python and it is capable of numerous mathematical operations. When you import Numpy as ‘np’, you unlock the full potential of Numpy.The benefit of using Import Numpy as np over regular python functions is that it offers extremely fast and efficient operations due to precompiled libraries written in C. This allows you to perform common mathematical operations such as array functions, linear algebra support, random number generation, polynomials and Fourier transforms much faster than regular Python functions."

# Tokenize and vectorize the documents
vectorizer = CountVectorizer()
X = vectorizer.fit_transform([document1, document2])

# Convert the sparse matrix to dense matrix
dense_matrix = X.toarray()

# Calculate cosine similarity using numpy
similarity = cosine_similarity(dense_matrix)

# The similarity matrix is symmetric, and the diagonal elements are 1 (documents are identical)
# The off-diagonal elements represent the cosine similarity between documents
print("Cosine Similarity Matrix:")
print(similarity)
