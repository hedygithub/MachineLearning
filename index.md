## Machine Learning

## Linear Regression
### Basic Concepts 
![](https://github.com/hedygithub/MachineLearning/blob/gh-pages/images/def_simple_linear_model.png)
![](https://github.com/hedygithub/MachineLearning/blob/gh-pages/images/def_general_linear_model.png)
- Xs are the covariates (or features, or inputs, or independent variables) 
- Y is the response (or outcomes, or outputs, or dependent variable.
- Noise Term (or errors): i.i.d. Gaussian random variables
- Residuals: The errors in our predictions

### Assumptions
1. Linearity: 
    - There is a linear relationship between the covariates and the response. 
    - Linear relationship can be assessed with scatter plots.
2. **Normality Why**?: 
    - Variables follow a Gaussian Distribution.
    - Normality can be assessed with histograms. Normality can also be statistically tested, for example with the Kolmogorov-Smirnov test.
    - When the variable is not normally distributed a non-linear transformation like Log-transformation may fix this issue.
3. The Noise Term
    - The error term is assumed to be a random variable that has a mean of 0 and normally distributed (i.i.d. Gaussian random variables)
    - **Why** ?
    ![](https://github.com/hedygithub/MachineLearning/blob/gh-pages/images/why_linear_model_error_term_normal_distributed.png)
    - When the errors are not normally distributed, it is OK if we have enough data.
4. Homoscedasticity
    - The error term has onstant variance σ2 at every value of X. 
    - **Why**?
    ![](https://github.com/hedygithub/MachineLearning/blob/gh-pages/images/why_linear_model_homoscedastic.png)
    - There are tests and plots to determine homescedasticity. Residual plots, Levene's test, Barlett's test, and Goldfeld-Quandt Test.
    - In the heteroscedastic, we can use Weighted Least Squares (WLS) to transform the problem into the homoscedastic case.
 5. **Residuals Why**
    - Residuals are statistically independent, have uniform variance, are normally distributed
 6. Non-Collinearity (Xs is full column rank in Linear Algerba): 
    - Multicolinearity occurs when the independent variables are correlated with each other.
    - **Why** : Multicolinearity means Xs has no full column rank, and by rank-nullity theorem, ker(Xs) has non-zero values, we have no unique soulution.
    - This can be check by heap map of correlation.
 
 ### Solutions: Mean Squared Error (MSE)
 1. Simple Linear Model:
 ![](https://github.com/hedygithub/MachineLearning/blob/gh-pages/images/simple_linear_model_solutions.png)
 2. General Linear Model:
 ![](https://github.com/hedygithub/MachineLearning/blob/gh-pages/images/general_linear_model_solutions.png)
 3. Confidence Interval
 4. Predicted Interval
 
 ### Useful Points of Linear Model
 1. In simple linear model, the fitted line pass through the sample central point.
 2. In simple linear model, if X and Y are standardized, correlation is equal to slope. 
 3. Centerated Resduals. In linear model, sum of residual equal to zero.
 4. Reduced Variance: 
    - R-squared: proportion of variance explained by the ﬁt
 5. Leverage: Slope has the highest sensitivity to points furthest from the mean 
  
 ### Question 1: More feature, less features?
 ### Question 2: What if you duplicate all the data and do regression on the new data set?
 ### Question 3: What are the Advantages/Disadvantages of Linear regression?
    


## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/hedygithub/DiHe.github.io/edit/gh-pages/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/hedygithub/DiHe.github.io/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
