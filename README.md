## Final Project of CS5351 Software Engineering

**Team 8**

**Team Leader**

LI ZHIYU

**Members:** 

CHANG CHONGYU

CHEN DATANG

FENG ZIXIAO

LI GUOQIONG

SUN CHENGWEI 

YE KAI

ZONG WEIHANG

## Brief Description

In this project, we plan to develop a drag-and-drop, block-based machine learning tool that enables developers to apply machine learning techniques more quickly and easily.



### Project Background Introduction

With the rapid development of Machine Learning (ML), it has become an essential part of modern technology. However, many developers, especially those outside the field of data science, may feel overwhelmed by the complex mathematical theories, programming requirements, and algorithm intricacies. As a result, there is a growing need for a tool that allows developers to learn and apply machine learning techniques in a more intuitive and straightforward way.

To address this demand, we plan to develop a **drag-and-drop, block-based machine learning tool**. This tool will lower the learning curve for machine learning by utilizing a modular approach, enabling developers to build and test machine learning models without needing an in-depth understanding of the underlying details. At the same time, this tool will help developers quickly grasp the core concepts of machine learning and apply them to real-world projects.

To achieve this goal, we plan to perform secondary development based on Google’s open-source project Blockly, integrating it with machine learning frameworks to create a powerful and user-friendly tool. Blockly's modular and drag-and-drop programming features will serve as the core foundation of our tool.

---

### Simple Plan

1. **Requirement Analysis and Design**  
   - Identify the target audience (e.g., beginners, non-data-science developers).  
   - Define the core functionalities of the tool, such as data preprocessing, model selection, training, and testing.  
   - Design an intuitive, simple, and easy-to-use user interface.
2. **Technology Selection**  
   - Customize machine learning-related blocks (e.g., data loading blocks, model training blocks) based on Google’s **Blockly**.  
   - Integrate with popular machine learning frameworks, such as TensorFlow or pytorch.  
3. **Module Development**  
   - **Data Processing Module**: Includes features for data importing, cleaning, and feature engineering.  
   - **Model Construction Module**: Enables users to choose different machine learning models (e.g., linear regression, decision trees, neural networks).  
   - **Model Training and Testing Module**: Allows users to set training parameters via blocks and visualize model performance.
4. **Visualization Features**  
   - Provide real-time visual feedback, such as data distribution and model performance metrics (e.g., accuracy, loss function trends).  
   - Use chart libraries (e.g., Recharts) to display training results and model evaluation data.
5. **Testing and Optimization**  
   - Conduct testing with different user groups to ensure usability and stability.  
   - Optimize code performance to ensure smooth data processing and training.
6. **Documentation and Tutorials**  
   - Write comprehensive user guides and developer documentation.  
   - Provide sample projects and instructional content to help users get started quickly.

---

### Project Goals

The final tool will have the following features:
- **Ease of Use**: Build and train machine learning models with simple drag-and-drop blocks.  
- **Visualization**: Provide real-time displays of data and model states to help users intuitively understand the machine learning process.  
- **High Extensibility**: Support various machine learning algorithms and allow future customization and expansion.  
- **Educational Value**: Help users not only build models but also learn machine learning concepts during the process.

Through this plan, we aim to create a tool that is both an effective learning resource and a platform for quickly validating machine learning ideas.

### References

1. **Blockly GitHub Repository**  
   [https://github.com/google/blockly](https://github.com/google/blockly)  

   The official repository for Google’s Blockly project, which provides the foundation for modular and drag-and-drop programming.

2. **Mind+ with ml5.js KNN**  
   [https://mindplus.dfrobot.com.cn/ml5-knn](https://mindplus.dfrobot.com.cn/ml5-knn)  

   An example integrating ml5.js and KNN, demonstrating how to use drag-and-drop programming for machine learning.

3. **TensorFlow.js**  
   [https://www.tensorflow.org/js](https://www.tensorflow.org/js)  
   A library for running machine learning models in the browser, ideal for integration with Blockly.

4. **Scratch**  
   [https://scratch.mit.edu](https://scratch.mit.edu)  
   一个基于积木块的编程语言，适合初学者学习编程概念。

这些资源可以帮助你更好地理解如何将拖放式编程与机器学习结合。