import math, re, os
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from collections import Counter 

print("TensorFlow version:", tf.__version__)
print("Keras version:", tf.keras.__version__)

# Changed from tf.data.experimental.AUTOTUNE to tf.data.AUTOTUNE
AUTO = tf.data.AUTOTUNE

#####
# Pre
#####

IMAGE_SIZE = (224, 224) 
EPOCHS = 15
BATCH_SIZE = 32

TRAINING_FILENAMES = tf.io.gfile.glob('data/train/*.tfrec')
VALIDATION_FILENAMES = tf.io.gfile.glob('data/val/*.tfrec')
TEST_FILENAMES = tf.io.gfile.glob('data/test/*.tfrec') 

CLASSES = ['pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea', 'wild geranium', 'tiger lily', 
           'moon orchid', 'bird of paradise', 'monkshood', 'globe thistle', 'snapdragon', "colt's foot", 'king protea', 
           'spear thistle', 'yellow iris', 'globe-flower', 'purple coneflower', 'peruvian lily', 'balloon flower', 
           'giant white arum lily', 'fire lily', 'pincushion flower', 'fritillary', 'red ginger', 'grape hyacinth', 
           'corn poppy', 'prince of wales feathers', 'stemless gentian', 'artichoke', 'sweet william', 'carnation', 
           'garden phlox', 'love in the mist', 'cosmos', 'alpine sea holly', 'ruby-lipped cattleya', 'cape flower', 
           'great masterwort', 'siam tulip', 'lenten rose', 'barberton daisy', 'daffodil', 'sword lily', 'poinsettia', 
           'bolero deep blue', 'wallflower', 'marigold', 'buttercup', 'daisy', 'common dandelion', 'petunia', 'wild pansy', 
           'primula', 'sunflower', 'lilac hibiscus', 'bishop of llandaff', 'gaura', 'geranium', 'orange dahlia', 
           'pink-yellow dahlia', 'cautleya spicata', 'japanese anemone', 'black-eyed susan', 'silverbush', 'californian poppy', 
           'osteospermum', 'spring crocus', 'iris', 'windflower', 'tree poppy', 'gazania', 'azalea', 'water lily', 'rose', 
           'thorn apple', 'morning glory', 'passion flower', 'lotus', 'toad lily', 'anthurium', 'frangipani', 'clematis', 
           'hibiscus', 'columbine', 'desert-rose', 'tree mallow', 'magnolia', 'cyclamen ', 'watercress', 'canna lily', 
           'hippeastrum ', 'bee balm', 'pink quill', 'foxglove', 'bougainvillea', 'camellia', 'mallow', 'mexican petunia', 
           'bromelia', 'blanket flower', 'trumpet creeper', 'blackberry lily', 'common tulip', 'wild rose']

###########
### Functions
###########

def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)  # Resize to target size
    return image

def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "class": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    label = tf.cast(example['class'], tf.int32)
    return image, label

def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "id": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    idnum = example['id']
    return image, idnum

def load_dataset(filenames, labeled=True, ordered=False):
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)
    return dataset

def data_augment(image, label):
    # Enhanced augmentation
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    return image, label   

def get_training_dataset():
    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)
    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    return dataset

def get_validation_dataset(ordered=False, cache=True):
    dataset = load_dataset(VALIDATION_FILENAMES, labeled=True, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    if cache:
        dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO)
    return dataset

def get_test_dataset(ordered=False):
    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    return dataset

def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

def display_confusion_matrix(cmat, score, precision, recall, normalized=''):
    plt.figure(figsize=(15,15))
    ax = plt.gca()
    ax.matshow(cmat, cmap='Reds')
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, fontdict={'fontsize': 7})
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
    ax.set_yticks(range(len(CLASSES)))
    ax.set_yticklabels(CLASSES, fontdict={'fontsize': 7})
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    titlestring = ""
    if score is not None:
        titlestring += 'f1 = {:.3f} '.format(score)
    if precision is not None:
        titlestring += '\nprecision = {:.3f} '.format(precision)
    if recall is not None:
        titlestring += '\nrecall = {:.3f} '.format(recall)
    if len(titlestring) > 0:
        ax.text(101, 1, titlestring, fontdict={'fontsize': 18, 'horizontalalignment':'right', 'verticalalignment':'top', 'color':'#804040'})
    plt.savefig(f'confusion_matrix{normalized}.png')
    plt.close()

def analyze_distribution(dataset, name, augmented=False):
    labels = []
    for batch in dataset:
        images, batch_labels = batch
        labels.extend(batch_labels.numpy())
    
    counter = Counter(labels)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(*zip(*sorted(counter.items())))
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{int(bar.get_height())}', ha='center', va='bottom')
    
    title = f'{name} Class Distribution'
    if augmented:
        title += ' (After Augmentation)'
    else:
        title += ' (Original)'
    
    plt.title(title)
    plt.xlabel('Class Label')
    plt.ylabel('Count')
    plt.grid(axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    filename = f'{name.lower()}_distribution'
    if augmented:
        filename += '_augmented'
    plt.savefig(f'{filename}.png')
    plt.close()
    
    return counter

####
#####

NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
NUM_VALIDATION_IMAGES = count_data_items(VALIDATION_FILENAMES)
NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
VALIDATION_STEPS = -(-NUM_VALIDATION_IMAGES // BATCH_SIZE)
TEST_STEPS = -(-NUM_TEST_IMAGES // BATCH_SIZE)
print('Dataset: {} training images, {} validation images, {} unlabeled test images'.format(
    NUM_TRAINING_IMAGES, NUM_VALIDATION_IMAGES, NUM_TEST_IMAGES))

# Calculate class weights for imbalanced data
print("Calculating class weights...")
original_train_ds = load_dataset(TRAINING_FILENAMES, labeled=True, ordered=True)
original_train_ds = original_train_ds.batch(BATCH_SIZE)
original_dist = analyze_distribution(original_train_ds, "Train")

labels = []
for batch in original_train_ds:
    _, batch_labels = batch
    labels.extend(batch_labels.numpy())

# Update the class weights calculation to avoid division by zero
class_counts = np.bincount(labels)
total_samples = np.sum(class_counts)
# Convert to numpy floats first, then to native Python floats
class_weights = {
    i: float(np.float64(total_samples / (len(class_counts) * count))) if count > 0 else 1.0 
    for i, count in enumerate(class_counts)
}

# Create callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_sparse_categorical_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model.keras',  # Changed to .keras format
    monitor='val_sparse_categorical_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

# Use EfficientNetB0 as base model
base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(*IMAGE_SIZE, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)

# Freeze base model layers initially
base_model.trainable = False

# Create new model on top
inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
x = tf.keras.applications.efficientnet.preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(len(CLASSES), activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

# Compile with lower initial learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy']
)

model.summary()

# First train just the top layers
history = model.fit(
    get_training_dataset(),
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=10,  
    validation_data=get_validation_dataset(),
    validation_steps=VALIDATION_STEPS,
    class_weight=class_weights,
    callbacks=[early_stopping, model_checkpoint, lr_scheduler]
)

# Unfreeze some layers for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

# Recompile with lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy']
)

# Fine-tune the model
history_fine = model.fit(
    get_training_dataset(),
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    initial_epoch=history.epoch[-1],
    validation_data=get_validation_dataset(),
    validation_steps=VALIDATION_STEPS,
    class_weight=class_weights,
    callbacks=[early_stopping, model_checkpoint, lr_scheduler]
)

# Load best model
model = tf.keras.models.load_model('best_model.keras')

####
## graphs
###

# A grahp for loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'] + history_fine.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'] + history_fine.history['val_loss'], label='Validation Loss')
plt.title('Model Loss', pad=20)
plt.ylabel('Loss', labelpad=10)
plt.xlabel('Epoch', labelpad=10)
plt.legend()
plt.grid(True)
plt.xticks(rotation=45 if len(history.history['loss'] + history_fine.history['loss']) > 10 else 0)
plt.tight_layout()
plt.savefig('loss_curve.png', bbox_inches='tight', dpi=300)
plt.close()

# A graph for accuracy
plt.figure(figsize=(12, 6))
plt.plot(history.history['sparse_categorical_accuracy'] + history_fine.history['sparse_categorical_accuracy'], 
         label='Training Accuracy')
plt.plot(history.history['val_sparse_categorical_accuracy'] + history_fine.history['val_sparse_categorical_accuracy'], 
         label='Validation Accuracy')
plt.title('Model Accuracy', pad=20)
plt.ylabel('Accuracy', labelpad=10)
plt.xlabel('Epoch', labelpad=10)
plt.legend()
plt.grid(True)
plt.xticks(rotation=45 if len(history.history['sparse_categorical_accuracy'] + history_fine.history['sparse_categorical_accuracy']) > 10 else 0)
plt.tight_layout()
plt.savefig('accuracy_curve.png', bbox_inches='tight', dpi=300)
plt.close()

# confusion matrix
cmdataset = get_validation_dataset(ordered=True)
images_ds = cmdataset.map(lambda image, label: image)
labels_ds = cmdataset.map(lambda image, label: label).unbatch()
cm_correct_labels = next(iter(labels_ds.batch(NUM_VALIDATION_IMAGES))).numpy()
cm_probabilities = model.predict(images_ds, steps=VALIDATION_STEPS)
cm_predictions = np.argmax(cm_probabilities, axis=-1)

cmat = confusion_matrix(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)))
score = f1_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')
precision = precision_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')
recall = recall_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')
display_confusion_matrix(cmat, score, precision, recall)
# Normalize
cmat = (cmat.T / cmat.sum(axis=1)).T
display_confusion_matrix(cmat, score, precision, recall,'_normalized')
print('f1 score: {:.3f}, precision: {:.3f}, recall: {:.3f}'.format(score, precision, recall))

### save model
model.save('model.keras')
model.export('model') 

## test data
test_ds = get_test_dataset(ordered=True)
print('Computing predictions...')
test_images_ds = test_ds.map(lambda image, idnum: image)
probabilities = model.predict(test_images_ds, steps=TEST_STEPS)
predictions = np.argmax(probabilities, axis=-1)
print(predictions)

print('Generating submission.csv file...')
test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U')
np.savetxt('submission.csv', np.rec.fromarrays([test_ids, predictions]), fmt=['%s', '%d'], 
           delimiter=',', header='id,label', comments='')