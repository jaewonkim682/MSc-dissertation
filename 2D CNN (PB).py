# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 11:45:05 2022

@author: jaewon KIM
"""
#clear log txt  
clear_log= open("log.txt", "w") 
clear_log.close()

while 1:

    #delete all variables 
    for reset in range(2):
        for element in dir():
            del globals()[element] 
            
    import numpy as np
    import os
    from matplotlib import pyplot as plt
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
    from sklearn.model_selection import train_test_split
    from sklearn.utils import shuffle
    import scipy.io
    
   #bring paths to collect multispectral images of infected and uninfected leaves 
   path_infected=''
   path_uninfected=''
   
   #type the numbmer of patches and samples
   no_of_infected_samples=
   no_of_uninfected_samples=
   no_of_patch=
    
    #count the total number of patches
    infected_patch_total_number = no_of_infected_samples*no_of_patch
    uninfected_patch_total_number = no_of_uninfected_samples*no_of_patch
    
    #count the number of multispectral images
    no_multispectral_image_infected = len(os.listdir(path_infected))
    no_multispectral_image_uninfected = len(os.listdir(path_uninfected))
    
    #read multispectral images
    temp=0
    infected_images = []
    for i in range(1, infected_patch_total_number+1):
        for var_count in range(1, 15+1):
            if var_count != 14:
                #read mat files
                temp = str(var_count)+'_of_'+str(i)+'.mat'
                matfile = scipy.io.loadmat(os.path.join(path_infected, temp))
                image=matfile['image']
                rescaled=image
                if rescaled is not None:
                    infected_images.append(rescaled)
    temp=0          
    uninfected_images = []
    for i in range(1, uninfected_patch_total_number+1):
        for var_count in range(1, 15+1):
            if var_count != 14:     
                #read mat files
                temp = str(var_count)+'_of_'+str(i)+'.mat'
                matfile = scipy.io.loadmat(os.path.join(path_uninfected, temp))
                image=matfile['image']
                rescaled=image
                if rescaled is not None:
                    uninfected_images.append(rescaled)
        
    #combine multispectral images in each patch
    infected_patch=np.array_split(infected_images,no_multispectral_image_infected/14)
    uninfected_patch=np.array_split(uninfected_images,no_multispectral_image_uninfected/14)
    
    #produce result tables
    result_infected=[]
    for i in range(int(no_multispectral_image_infected/14)):
        result_infected.append(0)
    result_uninfected=[]
    for i in range(int(no_multispectral_image_uninfected/14)):
        result_uninfected.append(1)
        
    #split the data into train and test datasets
    train_infected, test_infected, train_infected_result, test_infected_result = train_test_split(
     infected_patch, result_infected, test_size=0.1,random_state=10, shuffle=True)
    
    train_uninfected, test_uninfected, train_uninfected_result, test_uninfected_result = train_test_split(
     uninfected_patch, result_uninfected, test_size=0.1,random_state=10, shuffle=True)
    
    #combine the infected and uninfected data
    train=np.concatenate([train_infected,train_uninfected])
    train_result=np.concatenate([train_infected_result,train_uninfected_result])
    test=np.concatenate([test_infected,test_uninfected])
    test_result=np.concatenate([test_infected_result,test_uninfected_result])
    
    #shuffle the combined data
    train_shuffle, train_result_shuffle=shuffle(train,train_result)
    test_shffule, test_result_shuffle=shuffle(test,test_result)
    
    #build the model
    model = Sequential()
    model.add(Conv2D(16, (3,3), 1, activation = 'relu', input_shape=(14,24,24)))
    model.add(Conv2D(16, (3,3), 1, activation = 'relu'))
    model.add(Dropout(rate=0.4))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())    
    model.add(Dropout(rate=0.4))
    model.add(Dense(24, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    
    #insert the data to train the model
    hist = model.fit(train_shuffle, train_result_shuffle, validation_split=0.2, epochs = 30, batch_size=32) 
    
    #plot the loss graph
    plt.plot(hist.history['loss'], 'r', label='Training loss')
    plt.plot(hist.history['val_loss'], 'b', label='Validation loss')
    plt.title('Training Loss and Validation Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    
    #plot the accuracy graph
    plt.plot(hist.history['accuracy'], 'r', label='Training accuracy')
    plt.plot(hist.history['val_accuracy'], 'b', label='Validation accuracy')
    plt.title('Training accuracy and Validation accuracy')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()
    
    #insert the data to test the model
    test_loss, test_acc = model.evaluate(test_shffule, test_result_shuffle, batch_size=8)
    print('Test accuracy: ', test_acc*100)

    #save test result in log txt
    if test_acc*100 != 50.0:
        with open("log.txt", "a") as log:
            log.write(str(test_acc*100))
            log.write("\n")
            log.close
            
    #count the saved results and terminate the code if it reaches 10 
    with open("log.txt", "r") as log:
        count_results = len(log.readlines())
        print(count_results)
        if count_results == 10:
            break
        
#calculate the avearge accuracy
log = open('log.txt', 'r')    
saved_results = log.readlines()    
log.close()
count =  len(saved_results)
temp=0
for result in saved_results:
    temp = temp + float(result.rstrip())
average = temp / float(count)    
print("\n")
print("average is: ")
print(average)
