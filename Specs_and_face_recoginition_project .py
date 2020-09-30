#!/usr/bin/env python
# coding: utf-8

# **Face recognition with olivetti dataset**

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces
dataFace recognition with olivetti dataset

im
import numpy as np
import matplotlib.pyplot as plt
​ = fetch_olivetti_faces()
dir(data)


# In[ ]:


images = data.images
labels = data.target


images.shape, labels.shape


# In[ ]:


def show_image(images, label, how_many=5):


    ##set up figure size in inches
    fig=plt.figure(figsize=(12,12))import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces
dataFace recognition with olivetti dataset

im
import numpy as np
import matplotlib.pyplot as plt
​ = fetch_olivetti_faces()
dir(data)
    fig.subplots_adjust(left=0 , right=1 ,bottom=0, top=1, hspace=0.05,wspace=0.05)
    from random import randint

    for  i in range(how_many):
        #we will print images in matrix 10,10
        index = randint(0,images.shape[0])

        p = fig.add_subplot(10,10, i + 1 ,xticks=[],yticks=[])

        p.imshow( images[index], cmap=plt.cm.bone)
        #label the image with target value
        p.text(30, 14, str( label[index] ))
        p.text(30, 60, str(i))
        
show_image(images, labels, how_many=20)


# In[ ]:


from sklearn.svm import SVC
model_linear_kernel = SVC( kernel="linear")


# In[ ]:


images = images.reshape(-1, 64*64)

images = images / 255


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( images, labels, test_size=0.25, random_state=7)


# In[ ]:


model_linear_kernel.fit(X_train, y_train)


# In[ ]:


model_linear_kernel.score(X_test, y_test)


# In[ ]:


index_with_glasses = [
	(10, 19), (30, 32), (37, 38), (50, 59), (63, 64),
	(69, 69), (120, 121), (124, 129), (130, 139), (160, 161),
	(164, 169), (180, 182), (185, 185), (189, 189), (190, 192),
	(194, 194), (196, 199), (260, 269), (270, 279), (300, 309),
	(330, 339), (358, 359), (360, 369)
]


# In[ ]:


y = np.zeros(labels.shape)

for start, end in index_with_glasses:

    y[start: end + 1] = 1

#confirm for yourself
y


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split( images, y, test_size=0.25, random_state=7 )


# In[ ]:


model_linear_kernel = SVC(kernel = "linear" )
model_linear_kernel.fit(X_train, y_train)


# In[ ]:


show_image( X_test.reshape(-1,64,64), y_test )


# In[ ]:


model_linear_kernel.score(X_test, y_test)


# **Face recognition with lfw faces dataset**

# In[ ]:


from sklearn.datasets import fetch_lfw_people
data = fetch_lfw_people(min_faces_per_person=60)


# In[ ]:


images = data.images
labels = data.target

images.shape, labels.shape


# In[ ]:


data.target_names[6]


# In[ ]:


X = images.reshape(-1,62*47)
y = labels

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=7)


# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(n_components=15, whiten=True, random_state=7)


X_train_pca = pca.fit_transform(X_train)
X_test_pca  = pca.transform(X_test)


# In[ ]:


pca.explained_variance_ratio_[:].sum()


# In[ ]:


from sklearn.svm import SVC
model = SVC(kernel="rbf")


# In[ ]:


from sklearn.model_selection import GridSearchCV

param_grid = { 'C': [.01, 0.5, 1, 5, 10, 50], 'gamma': [0.0001, 0.0005, 0.001, 0.005] }

grid = GridSearchCV(model, param_grid)


# In[ ]:


get_ipython().run_line_magic('timeit', 'grid.fit(X_train_pca, y_train)')


# In[ ]:


grid.best_params_


# In[ ]:


best_model = grid.best_estimator_

predictions = best_model.predict(X_test_pca)
actuals      = y_test


# In[ ]:


import matplotlib.pyplot as plt

fig, ax = plt.subplots(4, 6)
for i, axi in enumerate(ax.flat):

    axi.imshow( X_test[i].reshape(62, 47), cmap='bone')
    axi.set(xticks=[], yticks=[])
    axi.set_ylabel( data.target_names[ predictions[i] ].split()[-1], color='black' if actuals[i] == predictions[i] else 'red')
    
fig.suptitle('Predicted Names; Incorrect Labels in Red', size=14);


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(actuals, predictions, target_names=data.target_names))


# In[ ]:





# In[ ]:




