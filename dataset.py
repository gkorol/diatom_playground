import os
import tensorflow as tf

dataset_dir = "/kaggle/input/diatom-dataset"
csv_dir = "/kaggle/input/diatom-csv/"
annotations_dir = os.path.join(dataset_dir, "annotations")
images_dir = os.path.join(dataset_dir, "images")
PATCH_SIZE = 256
NUM_PATCHES = 10

split_data = False
gen_patches = False

def save_images(images, loc):    
    pd.DataFrame(images).to_csv(loc + ".csv")
    

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = np.expand_dims(x, axis=-1)
    return x

def get_images(path):
    x_order = pd.read_csv(path)

    x = []
    y = []

    for idx, image in x_order.iterrows():
        x.append(read_image(os.path.join(images_dir, f"{image[1]}.png")))
        y.append(read_image(os.path.join(annotations_dir, f"{image[1]}.png")))

    return np.array(x), np.array(y)

if gen_patches:
    train_x, train_y = get_images(os.path.join(csv_dir, "train.csv"))
    val_x, val_y = get_images(os.path.join(csv_dir, "validation.csv"))

def get_norm_params(x):
    mean = np.mean(x)
    std = 0

    for im in x:
        std += ((im - mean) ** 2).sum() / (im.shape[0] * im.shape[1])

    std = np.sqrt(std / len(x))

    return mean, std

class DataGen(tf.keras.utils.Sequence):
  def __init__(self, dataset_dir, batch_size) :
        self.annotations_dir = os.path.join(dataset_dir, "annotations")
        self.images_dir = os.path.join(dataset_dir, "images")
        self.all_image_names = os.listdir(self.images_dir)
        self.batch_size = batch_size
    
  def __len__(self) :
        return (np.ceil(len(self.all_image_names) / float(self.batch_size))).astype(int)
  
  def __getitem__(self, idx):
        image_names = self.all_image_names[idx * self.batch_size : min((idx + 1) * self.batch_size, len(self.all_image_names) - 1)]
        
        x = []
        y = []

        for image in image_names:
            x.append(np.repeat(np.load(os.path.join(self.images_dir, image)), 3, -1))
            y.append(np.load(os.path.join(self.annotations_dir, image)) / 255)
       
        x = np.array(x)
        y = np.array(y)
        return x, y
    
  def __iter__(self):
    for idx in range(self.__len__()):
        yield self.__getitem__(idx)


def dice_coef(y_true, y_pred, smooth = 1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

def tversky(y_true, y_pred, smooth=1, alpha=0.7):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

def focal_tversky_loss(y_true, y_pred, gamma=1.5):
    y_true = tf.cast(y_true, tf.float32)
    tv = tversky(y_true, y_pred)
    return K.pow((1 - tv), gamma) 