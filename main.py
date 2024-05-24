import tensorflow as tf
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv 
import uuid

pos_path = os.path.join('data','positive')
neg_path = os.path.join('data','negative')
anc_path = os.path.join('data','anchor')

#Create dirs
def make_dirs(pos_path,neg_path,anc_path):
    os.makedirs(pos_path)
    os.makedirs(neg_path)
    os.makedirs(anc_path)

#make_dirs(pos_path, neg_path, anc_path)
'''
mv lfw files in negative dir

for directory in os.listdir('lfw'):
    for file in os.listdir(os.path.join('lfw',directory)):
        ex_path = os.path.join('lfw',directory,file)
        new_path = os.path.join(neg_path, file)
        os.replace(ex_path, new_path)
        
'''

'''Image Collection for positive and anchor classes'''
def collect_data(anc_path,pos_path,cam_index):
    cam_index = cam_index
    
    cap =cv.VideoCapture(cam_index)
    
    while cap.isOpened():
        ret,frame = cap.read()
        #Covnert frame 250x250
        frame = frame[120:120+250,200:200+250,:]
        cv.imshow('Image Collection',frame)
        
        #collect anchors
        if cv.waitKey(1) & 0XFF == ord('a'):
            img_name = os.path.join(anc_path,'{}.jpg'.format(uuid.uuid1()))
            cv.imwrite(img_name,frame)
        #collect positives
        if cv.waitKey(1) & 0XFF == ord('p'):
            img_name = os.path.join(pos_path,'{}.jpg'.format(uuid.uuid1()))
            cv.imwrite(img_name,frame)
        #Press q to exit
        if cv.waitKey(1) & 0XFF == ord('q'):
            print("[*] Exit")
            break
                
    cap.release()
    cv.destroyAllWindows()
        
'''
collect_data(anc_path, pos_path,1)
print(frame.shape)
plt.imshow(frame)
'''

'''Load and preprocess images'''
#Get image directories
anchor = tf.data.Dataset.list_files(anc_path+'\*.jpg').take(227)
positive =tf.data.Dataset.list_files(pos_path+'\*.jpg').take(227)
negative = tf.data.Dataset.list_files(neg_path+'\*.jpg').take(227)
#Convert to np iterator
dir_test = anchor.as_numpy_iterator()
print(dir_test.next())

#Preprocessing
def preprocess_images(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img,(105,105))
    img = img/255.0
    return img

img = preprocess_images('data\\anchor\\27ed3f3e-13c1-11ef-adf0-309c2381caac.jpg')
plt.imshow(img)

positivies = tf.data.Dataset.zip((anchor,positive,tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor,negative,tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positivies.concatenate(negatives)

samples = data.as_numpy_iterator()
example = samples.next()

def preprocess_twin(input_img,validation_img,label):
    return(preprocess_images(input_img),preprocess_images(validation_img),label)

res = preprocess_twin(*example)


data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=1024)
#Training partition
train_data = data.take(round(len(data)*.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)

train_samples = train_data.as_numpy_iterator()
train_sample = train_samples.next()

#Testting partition
test_data = data.skip(round(len(data)*.7))
test_data = test_data.take(round(len(data)*.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)

'''Model --> Siamese neural networks'''
#Embedding
def make_embedding():
    inputL = tf.keras.layers.Input(shape=(105,105,3),name='input_image') 
    
    c1 = tf.keras.layers.Conv2D(64,(10,10),activation='relu')(inputL)
    m1= tf.keras.layers.MaxPooling2D(64,(2,2),padding='same')(c1)
    
    c2 = tf.keras.layers.Conv2D(128,(7,7),activation='relu')(m1)
    m2 = tf.keras.layers.MaxPooling2D(64,(2,2),padding='same')(c2)
    
    c3 = tf.keras.layers.Conv2D(128,(4,4),activation='relu')(m2)
    m3 = tf.keras.layers.MaxPooling2D(64,(2,2),padding='same')(c3)
    
    c4 = tf.keras.layers.Conv2D(128,(4,4),activation='relu')(m3)
    f1 = tf.keras.layers.Flatten()(c4)
    d1 = tf.keras.layers.Dense(4096,activation='sigmoid')(f1)
    

    return tf.keras.models.Model(inputs=[inputL],outputs=[d1],name='embedding')

embedding = make_embedding()

print(embedding.summary())


class L1Dist(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super().__init__()
    
    def call(self,input_embedding,validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

l1 = L1Dist()

def make_siamese_model():
    #Anchor image input in the network
    input_image = tf.keras.layers.Input(name='input_image',shape=(105,105,3))
    
    #validation image in the network
    validation_image = tf.keras.layers.Input(name='validation_image',shape=(105,105,3))
    
    #combine siamese distance component
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image),embedding(validation_image))
    
    #Classification layer
    classifier = tf.keras.layers.Dense(1,activation='sigmoid')(distances)
    
    return tf.keras.models.Model(inputs=[input_image,validation_image],outputs=classifier,name='SiameseNetwork')

model = make_siamese_model()
print(model.summary())

'''Training'''
binary_cross_loss = tf.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)


checkpoint_dir = '/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir,'ckpt')
checkpoint = tf.train.Checkpoint(opt=optimizer,siamese_model=model)

#custom training
@tf.function
def train_step(batch):
    #record operations
    with tf.GradientTape() as tape:
        #Get anchor and positive/negative image
        X = batch[:2]
        #Get label
        y = batch[2]
        
        #Forward pass
        yhat = model(X,training=True)
        
        #Loss
        loss = binary_cross_loss(y,yhat)
    
    #Calculate gradient
    grad = tape.gradient(loss,model.trainable_variables)
    
    #Calculate updated weights and apply to model
    
    optimizer.apply_gradients(zip(grad,model.trainable_variables))
   
    #Return loss
    return loss

#Build training loop
def train(data, EPOCHS):
    for epoch in range(1, EPOCHS + 1):
        print('\n Epoch {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))
        for idx, batch in enumerate(data):
            train_step(batch)
            progbar.update(idx + 1)
        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

epochs = 10
train(train_data, epochs)

