# Customer Segmentation using K-Means Clustering

## Overview

This project implements **Customer Segmentation** using the **K-Means Clustering Algorithm**, an unsupervised machine learning technique. The goal is to group customers of a retail store into different segments based on their purchasing behavior, annual income, and spending patterns.

By identifying customer groups, businesses can better understand their customers and design targeted marketing strategies.

---

## Objective

To segment customers into meaningful groups using the K-Means clustering algorithm based on:

* Annual Income (k$)
* Spending Score (1-100)

---

## Dataset

The project uses the **Mall Customer Segmentation Dataset**.

### Dataset Features

| Feature                | Description                                        |
| ---------------------- | -------------------------------------------------- |
| CustomerID             | Unique customer identifier                         |
| Gender                 | Male/Female                                        |
| Age                    | Customer age                                       |
| Annual Income (k$)     | Annual income in thousand dollars                  |
| Spending Score (1-100) | Score assigned based on customer spending behavior |

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn

---

## Machine Learning Algorithm

### K-Means Clustering

K-Means is an unsupervised machine learning algorithm that groups similar data points into clusters.

#### Steps Performed

1. Load and explore the dataset.
2. Perform data visualization.
3. Select relevant features.
4. Determine the optimal number of clusters using the Elbow Method.
5. Apply K-Means Clustering.
6. Visualize customer segments.
7. Analyze and interpret clusters.

---

## Project Structure

```text
Customer-Segmentation-KMeans/
│
├── dataset/
│   └── Mall_Customers.csv
│
├── customer_segmentation.py
├── requirements.txt
├── customer_segments.csv(Generated as output file)
```

---

## Installation

### Clone the Repository

```bash
git clone <repository-url>
```

### Navigate to Project Folder

```bash
cd Customer-Segmentation-KMeans
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

Run the Python script:

```bash
python customer_segmentation.py
```

---

## Output

The project generates:

### Data Visualizations

* Age Distribution
* Annual Income Distribution
* Spending Score Distribution

### Elbow Method Graph

Used to determine the optimal number of clusters.

### Customer Segmentation Plot

Displays customer groups and cluster centroids.

### Output File

```text
customer_segments.csv
```

Contains the original dataset along with the assigned cluster for each customer.

---

## Results

The K-Means algorithm groups customers into multiple segments such as:

* High Income – High Spending Customers
* High Income – Low Spending Customers
* Low Income – High Spending Customers
* Low Income – Low Spending Customers
* Average Customers

These segments help businesses identify valuable customers and improve marketing strategies.

---


## Demo Video
https://drive.google.com/file/d/1IzdiMPYixsS7AeTvw4fC9C9MU4n0fJmO/view?usp=drive_link

## Learning Outcomes

Through this project, I gained practical experience in:

* Data Analysis
* Data Visualization
* Unsupervised Machine Learning
* K-Means Clustering
* Customer Segmentation
* Feature Selection
* Model Interpretation

---

## Future Improvements

* Add interactive visualizations.
* Deploy the project using Flask or Streamlit.
* Use additional customer features for advanced segmentation.
* Compare K-Means with other clustering algorithms.

---

## Author

**Nishitha Pallati**

Machine Learning Internship Project – Customer Segmentation using K-Means Clustering.
