<p align="right">
  Language Switch / 语言选择：
  <a href="./README.zh-CN.md">🇨🇳 中文</a> | <a href="./README.md">🇬🇧 English</a>
</p>

**INTRODUCTION**
---
This project is an interactive **Streamlit** application designed for learning and practicing
**Multi-Layer Perceptron (MLP)** fundamentals. Users can upload datasets, adjust model parameters, train the model,
and visualize training results and predictions in real time. The app is intended for students and beginners who want to
understand the basics of neural networks and how to build models for classification and regression problems. The
interactive interface helps users intuitively understand MLP structures, activation functions, loss functions, and
optimizers.

**DATA DESCRIPTION**
---
The application comes with built-in sample datasets suitable for supervised learning practice, including:

1. **Nonlinear Classification Dataset (2D)**
    - Features: two-dimensional coordinates `X1` and `X2`
    - Labels: classes `0` or `1`
    - Purpose: practice MLP classification tasks and observe the model's ability to fit nonlinear boundaries.

2. **Regression Dataset (optional)**
    - Features: one or more input dimensions `X`
    - Labels: continuous output `y`
    - Purpose: practice MLP regression tasks and explore how the model fits functions.

Users can also upload their own CSV datasets for training, as long as the format follows **feature columns + label
column** structure.

**FEATURES**
---

- **Data Loading & Visualization**: Upload CSV datasets and view them in tables and scatter plots.
- **Data Cleaning**: Automatically detect missing values and duplicate rows, with options to clean them.
- **Data Splitting**: Split datasets into training and testing sets with customizable test size and random seed.
- **Model Training**: Build and train an MLP model with configurable batch size and epochs.
- **Training Monitoring**: Track loss, accuracy, precision, recall, and AUC metrics in real-time.
- **Model Testing**: Predict using the trained model and display accuracy, R², MSE, MAE, and ROC curve.
- **Decision Boundary Visualization**: Dynamically show the decision boundary on scatter plots.

**WEB DEVELOPMENT**
---

1. Install NiceGUI with the command `pip install streamlit`.
2. Run the command `pip show streamlit` or `pip show streamlit | grep Version` to check whether the package has been
   installed and its version.
3. Run the command `streamlit run app.py` to start the web application.

**PRIVACY NOTICE**
---
This application may require inputting personal information or private data to generate customised suggestions,
recommendations, and necessary results. However, please rest assured that the application does **NOT** collect, store,
or transmit your personal information. All processing occurs locally in the browser or runtime environment, and **NO**
data is sent to any external server or third-party service. The entire codebase is open and transparent — you are
welcome to review the code [here](./) at any time to verify how your data is handled.

**LICENCE**
---
This application is licensed under the [BSD-3-Clause License](LICENSE). You can click the link to read the licence.

**CHANGELOG**
---
This guide outlines the steps to automatically generate and maintain a project changelog using git-changelog.

1. Install the required dependencies with the command `pip install git-changelog`.
2. Run the command `pip show git-changelog` or `pip show git-changelog | grep Version` to check whether the changelog
   package has been installed and its version.
3. Prepare the configuration file of `pyproject.toml` at the root of the file.
4. The changelog style is [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).
5. Run the command `git-changelog`, creating the `Changelog.md` file.
6. Add the file `Changelog.md` to version control with the command `git add Changelog.md` or using the UI interface.
7. Run the command `git-changelog --output CHANGELOG.md` committing the changes and updating the changelog.
8. Push the changes to the remote repository with the command `git push origin main` or using the UI interface.