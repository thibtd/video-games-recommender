# Video game recommender

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://tspgreedy.streamlit.app/) [![Makefile CI](https://github.com/thibtd/video-games-recommender/actions/workflows/MakefileCI.yml/badge.svg)](https://github.com/thibtd/video-games-recommender/actions/workflows/MakefileCI.yml)


### Summary of the App

#### Purpose
The Video Games Recommender app is designed to help users discover new video games based on their preferences. By leveraging machine learning and natural language processing techniques, the app provides personalized game recommendations, making it easier for users to find games they will enjoy.

#### Features
- **Personalized Recommendations**: Users can input the name of a game they like, and the app will recommend similar games based on various attributes such as genre, plot, and ratings.
- **Data-Driven Insights**: The app uses a combination of TF-IDF vectorization and a recommendation engine to analyze game descriptions and user ratings, ensuring accurate and relevant recommendations.
- **User-Friendly Interface**: Built with Streamlit, the app offers an intuitive and interactive user interface, making it easy for users to get recommendations quickly.

#### Strengths
- **Accurate Recommendations**: By using advanced machine learning algorithms and a well-curated dataset, the app provides highly accurate and relevant game recommendations.
- **Scalability**: The modular design of the app allows for easy updates and scalability, enabling the addition of new features and improvements over time.
- **Open Source**: The app is open-source, encouraging contributions from the community to enhance its functionality and accuracy.
- **Ease of Use**: The user-friendly interface ensures that even non-technical users can easily navigate the app and get the recommendations they need.
- **MLOps Integration**: The app incorporates MLOps practices to streamline the deployment and monitoring of machine learning models, ensuring robust and reliable performance.
- **CI/CD with GitHub Actions**: Continuous Integration and Continuous Deployment (CI/CD) pipelines are set up using GitHub Actions to automate testing, linting, and deployment processes, ensuring code quality and faster release cycles.

### Example Usage
1. **Input Game Name**: Users enter the name of a game they like.
2. **Get Recommendations**: The app processes the input and provides a list of similar games with relevant details such as ratings, genres, and plot summaries.
3. **Explore New Games**: Users can explore the recommended games and find new titles to enjoy.

### Technologies Used
- **Python**: The core programming language used for the app.
- **Streamlit**: For building the interactive user interface.
- **Pandas**: For data manipulation and analysis.
- **Scikit-learn**: For machine learning algorithms and vectorization.
- **NumPy**: For numerical computations.
- **GitHub Actions**: For setting up CI/CD pipelines to automate testing, linting, and deployment.
