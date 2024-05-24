import tensorflow as tf
import numpy as np
import os
import cv2 as cv 
import uuid
import matplotlib.pyplot as plt

# Path definitions
pos_path = os.path.join('data', 'positive')
neg_path = os.path.join('data', 'negative')
anc_path = os.path.join('data', 'anchor')

# Create directories
def make_dirs():
    os.makedirs(pos_path, exist_ok=True)
    os.makedirs(neg_path, exist_ok=True)
    os.makedirs(anc_path, exist_ok=True)

make_dirs()

mv lfw files in negative dir

for directory in os.listdir('lfw'):
    for file in os.listdir(os.path.join('lfw',directory)):
        ex_path = os.path.join('lfw',directory,file)
        new_path = os.path.join(neg_path, file)
        os.replace(ex_path, new_path)
        


# Image Collection function
def collect_data(anc_path, pos_path, cam_index=0):
    cap = cv.VideoCapture(cam_index)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame[120:120+250, 200:200+250, :]
        cv.imshow('Image Collection', frame)
        
        if cv.waitKey(1) & 0xFF == ord('a'):
            img_name = os.path.join(anc_path, f'{uuid.uuid1()}.jpg')
            cv.imwrite(img_name, frame)
        
        if cv.waitKey(1) & 0xFF == ord('p'):
            img_name = os.path.join(pos_path, f'{uuid.uuid1()}.jpg')
            cv.imwrite(img_name, frame)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
                
    cap.release()
    cv.destroyAllWindows()

# Preprocess images
def preprocess_images(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (105, 105))
    img = img / 255.0
    return img

# Embedding network
def make_embedding():
    inputL = tf.keras.layers.Input(shape=(105, 105, 3), name='input_image')
    x = tf.keras.layers.Conv2D(64, (10, 10), activation='relu')(inputL)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, (7, 7), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, (4, 4), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(256, (4, 4), activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(4096, activation='sigmoid')(x)
    return tf.keras.Model(inputs=[inputL], outputs=[x], name='embedding')

embedding = make_embedding()

# L1 Distance layer
class L1Dist(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(L1Dist, self).__init__(**kwargs)
    
    def call(self, input_embedding, validation_embedding):
        # Ensure that both embeddings are tensors
        input_embedding = tf.convert_to_tensor(input_embedding)
        validation_embedding = tf.convert_to_tensor(validation_embedding)
        return tf.math.abs(input_embedding - validation_embedding)

l1 = L1Dist()

# Siamese model
def make_siamese_model():
    input_image = tf.keras.layers.Input(name='input_image', shape=(105, 105, 3))
    validation_image = tf.keras.layers.Input(name='validation_image', shape=(105, 105, 3))
    
    distances = L1Dist()(embedding(input_image), embedding(validation_image))
    
    classifier = tf.keras.layers.Dense(1, activation='sigmoid')(distances)
    
    return tf.keras.Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')

model = make_siamese_model()
model.summary()

# Prepare data
anchor = tf.data.Dataset.list_files(anc_path + '/*.jpg').take(227)
positive = tf.data.Dataset.list_files(pos_path + '/*.jpg').take(227)
negative = tf.data.Dataset.list_files(neg_path + '/*.jpg').take(227)

positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)

def preprocess_twin(input_img, validation_img, label):
    return (preprocess_images(input_img), preprocess_images(validation_img), label)

data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=1024)

train_data = data.take(round(len(data) * 0.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)

test_data = data.skip(round(len(data) * 0.7))
test_data = test_data.take(round(len(data) * 0.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)

# Training
binary_cross_loss = tf.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
checkpoint_dir = os.path.abspath('./training_checkpoints')
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(optimizer=optimizer, siamese_model=model)

@tf.function
def train_step(batch):
    with tf.GradientTape() as tape:
        X = batch[:2]
        y = batch[2]
        yhat = model(X, training=True)
        
        # Ensure the shapes of y and yhat match
        y = tf.expand_dims(y, axis=-1)  # Convert y to shape (16, 1)
        yhat = tf.squeeze(yhat, axis=-1)  # Convert yhat to shape (16,)
        yhat = tf.transpose(yhat)  # Transpose yhat to shape (16, 1)
        
        loss = binary_cross_loss(y, yhat)
    grad = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grad, model.trainable_variables))
    return loss

def train(data, EPOCHS):
    for epoch in range(1, EPOCHS + 1):
        print('\n Epoch {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))
        for idx, batch in enumerate(data):
            train_step(batch)
            progbar.update(idx + 1)
        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

epochs = 50
train(train_data, epochs)

# Model evaluation
test_input, test_val, y_true = test_data.as_numpy_iterator().next()
y_hat = model.predict([test_input, test_val])

# Calculate predictions
y_hat_labels = np.where(y_hat > 0.5, 1, 0)

# Calculate recall
recall = tf.keras.metrics.Recall()
recall.update_state(y_true, y_hat_labels)
print("Recall:", recall.result().numpy())

# Save model
model.save('siamese_model.h5')

#Reload model

model = tf.keras.models.load_model('siamese_model.h5', custom_objects={
    'L1Dist': L1Dist, 'BinaryCrossentropy': tf.keras.losses.BinaryCrossentropy
})

y_pred = model.predict([test_input,test_val])

'''Real Time Test'''
def verify(frame,model,detection_hold,verification_hol):
    
    results = []
    for image in os.listdir(os.path.join('application_data','verification_image')):
        input_img = preprocess_images(os.path.join('application_data','input_image','input_image.jpg'))
        validation_img = preprocess_images(os.path.join('application_data','verification_image',image))
        
        result = model.predict(list(np.expand_dims([input_img,validation_img],axis=1)))
        results.append(result)
    
    detection = np.sum(np.array(results) > detection_hold)
    verification = detection / len(os.listdir(os.path.join('application_data','verification_image')))
    verified = verification > verification_hol
    
    return results,verified

cap = cv.VideoCapture(1)
while cap.isOpened():
    ret,frame = cap.read()
    frame = frame[120:120+250,200:200+250,:]
    cv.imshow('Verification',frame)
    if cv.waitKey(10) & 0xFF == ord('v'):
        #Write image to input_image path
        cv.imwrite(os.path.join('application_data','input_image','input_image.jpg'),frame)
        #Run verification
        results,verified = verify(frame,model,0.9,0.7)
        print(verified)
    if cv.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
    
