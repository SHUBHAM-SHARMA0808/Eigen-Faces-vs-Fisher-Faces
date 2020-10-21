import numpy as np 
import cv2
import os
import sys
from PIL import Image



def main():
    #Data preprocessing
    '''for i in range(50):
        print(i+1)
        for j in range(9):
            print(j+1)
            path = 'dataset/train_data/{}-0{}.jpg'.format(i+1, j+1)
            face = cv2.imread(path, 0)
            print(face)
            #normalized = face.gray().scale(100, 100)
            face_path = 'training_images/subject{}'.format(i+1)
            #ensure_dir_exists(face_path)
            sv_path = '{}/{}.jpg'.format(face_path, j + 1)
            print(sv_path)
            cv2.imwrite(sv_path, face)
            #normalized.save_to('{}/{}.png'.format(face_path, j + 1))'''

    list_of_arrays_of_images, labels_list, \
    list_of_matrices_of_flattened_class_samples = \
        read_images('training_images')
    #print("*")
    #print(labels_list)
    #print("*")
    '''print(list_of_arrays_of_images)
    print(" ")
    print(labels_list)
    print(" ")
    print(list_of_matrices_of_flattened_class_samples)'''    

    # create matrix to store all flattened images
    images_matrix = np.array([np.array(Image.fromarray(img)).flatten()
    for img in list_of_arrays_of_images],'f')

    # perform PCA
    eigenfaces_matrix, variance, mean_image = pca(images_matrix)

    projected_classes = []
    # Projecting each class sample (as class matrix) and then using the class average as the class weights for comparison with the Target image
    for class_sample in list_of_matrices_of_flattened_class_samples:
        class_weights_vertex = project_image(class_sample, mean_image, eigenfaces_matrix)
        projected_classes.append(class_weights_vertex.mean(0)) 
    
    '''count = 0
    for i in range(11):
        #print(i+1)
        for j in range(1):
            #print(j+1)
            path = 'test_data/subject{}/{}.jpg'.format(i+1, j+1)
            face = cv2.imread(path, 0) 
            flatten_face = np.array(face.flatten())
            predicted_class = predict_face(face, mean_image, eigenfaces_matrix)
            if(predicted_class == i+1):
                count = count+1 
    efficiency = (count*100)/10
    print("efficiency = {}".format(efficiency))'''

    list_of_arrays_of_test_images, test_labels_list, \
    list_of_matrices_of_flattened_test_class_samples = \
        read_images('test_data')     
    
    i = 0
    count = 0
    for timg in list_of_matrices_of_flattened_test_class_samples:
        predicted_class = predict_face(timg, mean_image, eigenfaces_matrix, projected_classes, labels_list)
        if(predicted_class == test_labels_list[i]):
            count = count+1                
        i = i+1
    print("\nTrined on {} images.\n".format(len(list_of_arrays_of_images)))
    print("Total {} images tested\n".format(i))
    print("{} matched correctly\n".format(count))    
    accuracy = (count*100)/i
    print("efficiency = {}\n".format(accuracy))       

def project_image(X, mean_image, eigenfaces_matrix):
    X = X - mean_image
    return np.dot(X, eigenfaces_matrix.T)

def predict_face(X, mean_image, eigenfaces_matrix, projected_classes, labels_list):
    min_class = -1
    min_distance = np.finfo('float').max
    projected_target = project_image(X, mean_image, eigenfaces_matrix)
    # delete last array item, it's nan
    projected_target = np.delete(projected_target, -1)
    for i in range(len(projected_classes)):
        distance = np.linalg.norm(projected_target - np.delete(projected_classes[i], -1))
        if distance < min_distance:
            min_distance = distance
            min_class = labels_list[i]
    # print(min_class, min_distance)
    return min_class    

def read_images(path, sz=None):
    """Reads the images in a given folder, resizes images on the fly if size is given.

    Args:
        path: Path to a folder with subfolders representing the subjects (persons).
        sz: A tuple with the size Resizes

    Returns:
        A tuple of (images, image_labels, class_matrix_list) where
            images: The images, which is a Python list of numpy arrays.
            image_labels: The corresponding labels (the unique number of
            the subject, person).
    """
    class_samples_list = []
    class_matrices_list = []
    images, image_labels = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            class_samples_list = []
            for filename in os.listdir(subject_path):
                if filename != ".DS_Store":
                    try:
                        im = Image.open(os.path.join(subject_path, filename))
                        # resize to given size (if given) e.g., sz = (480, 640)
                        if (sz is not None):
                            im = im.resize(sz, Image.ANTIALIAS)
                        images.append(np.asarray(im, dtype = np.uint8))

                    except IOError as e:
                        errno, strerror = e.args
                        print("I/O error({0}): {1}".format(errno, strerror))
                    except:
                        print("Unexpected error:", sys.exc_info()[0])
                        raise
                    # adds each sample within a class to this List
                    class_samples_list.append(np.asarray(im, dtype = np.uint8))

            # flattens each sample within a class and adds the array/vector to a class matrix
            class_samples_matrix = np.array([img.flatten()
                for img in class_samples_list],'f')

             # adds each class matrix to this MASTER List
            class_matrices_list.append(class_samples_matrix)

            image_labels.append(subdirname)

    return images, image_labels, class_matrices_list

def pca(X):
    """Principal Component Analysis

    input: X, matrix with training data stored as flattened arrays in rows
    return: projection matrix (with most important dimensions first).
    """
    # get dimensions
    num_data, dim = X.shape

    # center data
    mean_X = X.mean(axis=0)
    X = X - mean_X

    if dim > num_data:
        # PCA - compact trick used
        M = np.dot(X,X.T) # covariance matrix
        e,EV = np.linalg.eigh(M) # eigenvalues and eigenvectors
        #print(e)
        tmp = np.dot(X.T,EV).T # this is the compact trick
        V = tmp[::-1] # reverse since last eigenvectors are the ones we want
        S = np.sqrt(abs(e[::-1])) # reverse since eigenvalues are in increasing order

        for i in range(V.shape[1]):
            V[:,i] /= S
    else:
        # PCA - SVD used
        U, S, V = np.linalg.svd(X)
        V = V[:num_data-200] # only makes sense to return the first num_data

    # return the projection matrix, the variance and the mean
    return V, S, mean_X

main()
        