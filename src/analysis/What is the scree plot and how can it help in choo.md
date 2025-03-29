<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# What is the scree plot and how can it help in choosing the number of components

---
A **scree plot** is a graphical representation used in Principal Component Analysis (PCA) to help determine the optimal number of principal components to retain. It displays the eigenvalues associated with each principal component, typically arranged in descending order. The x-axis represents the principal components, while the y-axis shows their corresponding eigenvalues or explained variance.

## How the Scree Plot Helps

1. **Visualizing Variance Explained:**
The scree plot illustrates how much variance each principal component captures. Generally, the first few components will explain a significant portion of the total variance, while subsequent components contribute less.
2. **Identifying the "Elbow":**
One common method for selecting the number of components is to look for an "elbow" in the plot. The elbow point indicates where adding more components yields diminishing returns in explained variance. Retaining components before this point helps maintain most of the informative content without unnecessary complexity[^1][^5].
3. **Kaiser’s Rule:**
Another approach is Kaiser’s rule, which suggests retaining components with eigenvalues greater than 1. This criterion can also be visually represented on the scree plot as a horizontal line at 1, helping to identify significant components[^1][^2].
4. **Cumulative Explained Variance:**
Some scree plots also show cumulative explained variance, allowing you to see how many components are needed to reach a desired threshold (e.g., 80% or 90% of total variance) and aiding in decision-making regarding component retention[^4].

## Summary

In summary, a scree plot is a valuable tool for determining how many principal components to keep in PCA by visualizing explained variance and identifying key points like elbows or thresholds based on eigenvalues. By using this method, you can make informed decisions about dimensionality reduction while preserving essential information from your dataset.

<div style="text-align: center">⁂</div>

[^1]: https://statisticsglobe.com/scree-plot-pca

[^2]: https://www.graphpad.com/guides/prism/latest/statistics/stat_pca_example_scree_plot.htm

[^3]: https://blogs.sas.com/content/iml/2017/08/02/retain-principal-components.html

[^4]: https://analyse-it.com/docs/user-guide/multivariate/scree-plot

[^5]: https://en.wikipedia.org/wiki/Scree_plot

[^6]: https://web.pdx.edu/~newsomj/semclass/ho_efa.pdf

[^7]: https://www.sciencedirect.com/topics/mathematics/scree-plot

[^8]: https://pypi.org/project/screeplot/

