# What-If: Explainable AI Image Counterfactual Generator
## Introduction
What-If is an interactive web application that generates counterfactual explanations for image classification models. It allows users to explore alternative scenarios and outcomes by generating new images that would lead to a different classification.

## Features
- Counterfactual Generation: Generate new images that would lead to a different classification
- Image Classification: Classify images using a pre-trained model
- Explainability: Provide insights into the decision-making process of the model
- Interactive Interface: User-friendly interface for selecting target classes and generating counterfactuals
  
## How it Works
- Image Upload: Upload an image to the application
- Image Classification: Classify the image using a pre-trained model
- Target Class Selection: Select a target class to generate a counterfactual for
- Counterfactual Generation: Generate a new image that would lead to the selected target class
- Comparison: Compare the original image with the generated counterfactual
  
## Technologies Used
- Python: Programming language used for development
- Streamlit: Framework used for building the web application
- PyTorch: Deep learning framework used for image classification and counterfactual generation
- OpenCV: Library used for image processing
  
## Installation
To run the application, follow these steps:

- Clone the repository: ```git clone https://github.com/Gangadhar24377/what-if.git```
- Change the directory: ```cd what-if```
- Install the required dependencies: ```pip install -r requirements.txt```
- Run the application: ```streamlit run app.py```

# Examples:
![image](https://github.com/user-attachments/assets/df929a8c-9321-43f7-b4ba-93d19bd7aaf9)

## Areas of Significant Change
![image](https://github.com/user-attachments/assets/7ced1157-dbdd-467a-b7d7-3385425004da)




## Acknowledgments
This project was inspired by the concept of counterfactual explanations in explainable AI. The application uses pre-trained models and libraries to provide a user-friendly interface for generating counterfactuals.

## Note
This is a basic implementation of a counterfactual generator, and the accuracy of the generated counterfactuals may not be optimal. This is an ongoing project, and we plan to improve the accuracy and functionality of the application in future updates. If you have any suggestions or contributions, please feel free to contact.

# Disclaimer
Please note that the accuracy of the generated counterfactuals may vary depending on the quality of the input image and the pre-trained model used. This application is intended for educational and research purposes only, and should not be used for any critical or high-stakes applications.
