import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import datetime as dt
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

data = pd.read_csv('Online_Retail.csv');
data.head()

data.shape
data.info()

df_null = round(100*(data.isnull().sum())/len(data), 2)
df_null

data = data.dropna()
data['CustomerID'] = data['CustomerID'].astype(str)


#Monetary
data['Amount'] = data['Quantity']*data['UnitPrice']
mt_rfm = data.groupby('CustomerID')['Amount'].sum()
mt_rfm = mt_rfm.reset_index()
mt_rfm.head()

#Frequency
f_rfm = data.groupby('CustomerID')['InvoiceNo'].count()
f_rfm = f_rfm.reset_index()
f_rfm.columns = ['CustomerID','Frequency']
f_rfm.head()

#Merge the two dataframes (df) of Monetary and Frequency
rfm = pd.merge(mt_rfm, f_rfm , on='CustomerID', how='inner')
rfm.head()

#Recency
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

date_max = max(data['InvoiceDate'])
date_max

data['Difference'] = date_max - data['InvoiceDate']
data.head()


r_rfm = data.groupby('CustomerID')['Difference'].min()
r_rfm = r_rfm.reset_index()
r_rfm.head()

r_rfm['Difference'] = r_rfm['Difference'].dt.days
r_rfm.head()

rfm = pd.merge(rfm, r_rfm, on='CustomerID', how='inner')
rfm.columns = ['CustomerID' , 'Amount' , 'Frequency' , 'Recency']
rfm.head()


attributes = ['Amount', 'Frequency', 'Recency']
plt.rcParams['figure.figsize'] = [10,8]
sns.boxplot(data = rfm[attributes], orient="v", palette="Set2", whis=1.5, saturation=1, width=0.7)
plt.title("Outliers Variable Distribution", fontsize = 16, fontweight = 'bold')
plt.ylabel("Range") 
plt.xlabel("Attributes")

# Removing outliers
#Amount
Q1 = rfm.Amount.quantile(0.05)
Q2 = rfm.Amount.quantile(0.95)

IQR = Q2 - Q1
rfm= rfm[(rfm.Amount >= Q1 - 1.5*IQR) & (rfm.Amount <= Q2 + 1.5*IQR)]

#Recency
Q1 = rfm.Recency.quantile(0.05)
Q2 = rfm.Recency.quantile(0.95)

IQR = Q2 - Q1
rfm = rfm[(rfm.Recency >= Q1 - 1.5*IQR) & (rfm.Recency <= Q2 + 1.5*IQR)]

#Frequency
Q1 = rfm.Frequency.quantile(0.05)
Q2 = rfm.Frequency.quantile(0.95)

IQR = Q2 - Q1
rfm = rfm[(rfm.Frequency >= Q1 - 1.5*IQR) & (rfm.Frequency <= Q2 + 1.5*IQR)]

#Scaling of the Features

df_rfm = rfm[['Amount','Frequency','Recency']]

scaler = StandardScaler()

#Fit Transform
df_rfm_scaled = scaler.fit_transform(df_rfm)
df_rfm_scaled.shape

df_rfm_scaled = pd.DataFrame(df_rfm_scaled)
df_rfm_scaled.columns = ['Amount', 'Frequency', 'Recency']

#MODEL BUILDING
#k-means
kmeans = KMeans(n_clusters = 4, max_iter = 50)
kmeans.fit(df_rfm_scaled)

kmeans.labels_
set(kmeans.labels_)

sd = []
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]

for num_clusters in range_n_clusters: 
    kmeans = KMeans(n_clusters = num_clusters, max_iter=50) 
    kmeans.fit(df_rfm_scaled)

    sd.append(kmeans.inertia_)

# plot the SDs for each n_clusters 
plt.plot(sd)

kmeans = KMeans(n_clusters = 3, max_iter = 300)
kmeans.fit(df_rfm_scaled)


kmeans.labels_

rfm['Cluster_Id'] = kmeans.labels_
rfm.head()


sns.stripplot(x='Cluster_Id', y='Amount', data=rfm)

sns.stripplot(x='Cluster_Id', y='Frequency', data=rfm)

sns.stripplot(x='Cluster_Id', y='Recency', data=rfm)



import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk


# Load data
data = pd.read_csv('Online_Retail.csv')

# Function to perform RFM analysis
# Function to perform RFM analysis
def perform_rfm_analysis():
    global data
    global rfm
    
    # Compute monetary
    data['Amount'] = data['Quantity'] * data['UnitPrice']
    mt_rfm = data.groupby('CustomerID')['Amount'].sum().reset_index()

    # Compute frequency
    f_rfm = data.groupby('CustomerID')['InvoiceNo'].count().reset_index()
    f_rfm.columns = ['CustomerID', 'Frequency']

    # Compute recency
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
    date_max = max(data['InvoiceDate'])
    data['Difference'] = date_max - data['InvoiceDate']
    r_rfm = data.groupby('CustomerID')['Difference'].min().reset_index()
    r_rfm['Difference'] = r_rfm['Difference'].dt.days

    # Merge RFM data
    rfm = pd.merge(mt_rfm, f_rfm, on='CustomerID', how='inner')
    rfm = pd.merge(rfm, r_rfm, on='CustomerID', how='inner')
    rfm.columns = ['CustomerID', 'Amount', 'Frequency', 'Recency']

    # Display success message
    messagebox.showinfo("RFM Analysis", "RFM analysis has been performed successfully!")



# Function to display RFM results on GUI
def display_rfm_results():
    global rfm

    # Create a new Tkinter window for displaying RFM results
    rfm_window = tk.Toplevel()
    rfm_window.title("RFM Results")

    # Create a treeview to display RFM data
    rfm_tree = ttk.Treeview(rfm_window)
    rfm_tree["columns"] = ("Amount", "Frequency", "Recency")
    rfm_tree.heading("#0", text="CustomerID")
    rfm_tree.heading("Amount", text="Amount")
    rfm_tree.heading("Frequency", text="Frequency")
    rfm_tree.heading("Recency", text="Recency")

    # Insert RFM data into the treeview
    for index, row in rfm.iterrows():
        rfm_tree.insert("", "end", text=row['CustomerID'], values=(row['Amount'], row['Frequency'], row['Recency']))

    # Pack the treeview
    rfm_tree.pack(expand=True, fill="both")

    # Display success message
    messagebox.showinfo("RFM Results", "RFM results have been displayed successfully!")


# Function to perform K-means clustering
def perform_kmeans_clustering():
    global rfm

    # Selecting features for clustering
    features = ['Amount', 'Frequency', 'Recency']
    df_rfm = rfm[features]

    # Scaling the features
    scaler = StandardScaler()
    df_rfm_scaled = scaler.fit_transform(df_rfm)

    # Applying K-means clustering
    kmeans = KMeans(n_clusters=4, max_iter=50)
    kmeans.fit(df_rfm_scaled)

    # Assigning cluster labels
    rfm['Cluster_Id'] = kmeans.labels_

    # Display success message
    messagebox.showinfo("Clustering", "K-means clustering has been performed successfully!")



# Function to display clustering results on GUI
def display_clustering_results():
    global rfm

    # Create a new Tkinter window for displaying clustering results
    clustering_window = tk.Toplevel()
    clustering_window.title("Clustering Results")

    # Create a scatter plot for clustering results
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot clusters
    sns.scatterplot(x='Amount', y='Frequency', hue='Cluster_Id', data=rfm, palette='Set1', ax=ax)

    # Set plot title and labels
    plt.title('Clustering Results')
    plt.xlabel('Amount')
    plt.ylabel('Frequency')

    # Embedding the plot in the GUI
    canvas = FigureCanvasTkAgg(fig, master=clustering_window)
    canvas.draw()
    canvas.get_tk_widget().pack()

    # Display success message
    messagebox.showinfo("Clustering Results", "Clustering results have been displayed successfully!")

# Create Tkinter window
root = tk.Tk()
root.title("Customer Segmentation Analysis")

root.geometry("800x600")

background_image = Image.open("bgimage.jpg")
background_photo = ImageTk.PhotoImage(background_image)

# Create a label with the background image
background_label = tk.Label(root, image=background_photo)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

#Text at the top of the window
title_label = ttk.Label(root, text="Customer Segmentation Analysis", font=("Helvetica", 16, "bold"))
title_label.pack(side="top", fill="x")

# Create a frame to contain the buttons
button_frame = ttk.Frame(root)
button_frame.pack(side="left", padx=20, pady=20)

button_font = ('Helvetica', 12)
style = ttk.Style()
style.configure('Custom.TButton', font=button_font)

# Button to perform RFM analysis
rfm_button = ttk.Button(button_frame, text="Perform RFM Analysis", command=perform_rfm_analysis,width=30 , style='Custom.TButton')
rfm_button.pack(fill="x", pady=5)

# Button to display RFM results
rfm_results_button = ttk.Button(button_frame, text="Display RFM Results", command=display_rfm_results, width=30, style='Custom.TButton')
rfm_results_button.pack(fill="x", pady=5)

# Button to perform K-means clustering
clustering_button = ttk.Button(button_frame, text="Perform Clustering", command=perform_kmeans_clustering, width=30, style='Custom.TButton')
clustering_button.pack(fill="x", pady=5)

# Button to display clustering results
clustering_results_button = ttk.Button(button_frame, text="Display Clustering Results", command=display_clustering_results, width=30, style='Custom.TButton')
clustering_results_button.pack(fill="x", pady=5)


# Run the Tkinter event loop
root.mainloop()
