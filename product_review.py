import pandas as pd
import numpy as np
import os
import cv2
#product = {"Product_Name":["IPhone 15 Pro", " Samsung Refrigerator", "Oppo Reno 5G","American Tourist Bag"]}

#data = pd.read_csv("product_review.csv")
#data2 = pd.read_csv("image_path.csv")

#data.to_csv('product_review.csv',)
#data[data['Product_Name']=="IPhone 15 Pro"]['image_path'] = "C:/Users/krishnendu/VSCODE_projects/ML-recruitment-assessment/products/iphone15_pro"
#Review_customer = {"Product_Name":[],"Review":[],"Sentiment":[]}
#Review_customer = pd.DataFrame(Review_customer)

#Review_customer.loc[Review_customer.shape[0]] = {"Product_Name":"Iphone","Review":"good","Sentiment":"ok ok"}
image_path = pd.read_csv("image_path.csv")
"""print(image_path)
image_path = image_path.drop(image_path.index)
image_path = image_path.drop("Unnamed: 0",axis = 1)
#print(image_path)


folder_path = "products/products"
pathlist = os.listdir(folder_path)
product_names =  []
for name in pathlist:
    name = name.split(".")
    product_names.append(str(name[0]))
print(product_names)
#print(image_path)
image_extensions = ["png", "jpg", "jpeg", "gif"]  # Add other extensions if needed

image_files = [
    os.path.join(folder_path, filename)
    for filename in os.listdir(folder_path)
    if any(filename.lower().endswith(ext) for ext in image_extensions)
]
for i, row in enumerate(zip(product_names,image_files)):
    image_path.loc[i] = row
print(image_files)
print(image_path)
image_path.to_csv('image_Path.csv')"""

img = cv2.imread(image_path['image_path'][2])
img = cv2.resize(img,[80,40])
cv2.imshow('img',img)
cv2.waitKey(10000)
#print(data3)"""

