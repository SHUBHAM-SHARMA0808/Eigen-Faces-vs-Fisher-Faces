import cv2, sys, os
import numpy as np
import math

subjects = len(os.listdir('./dataset'))
print("Total subjects = {0}".format(subjects))
print("\n")

img = cv2.imread('./dataset/s1/1.pgm', cv2.IMREAD_GRAYSCALE)
print("checking img dim\n")
print(img)
 
images_per_subject = 10
train_imgs = 7
test_imgs = images_per_subject-train_imgs

rows, cols = img.shape

image_matrix_cols = train_imgs * subjects  # col# of image matrix
image_matrix_rows = rows * cols        # row# of image matrix 
#n = train_imgs * subjects  # col# of image matrix
#d = rows * cols        # row# of image matrix 
def train():
    ##### Forminig Image Matrix by adding images as col vector #####
    img_mat = np.empty((image_matrix_rows, image_matrix_cols))
    img_mat1 = np.empty((image_matrix_rows, image_matrix_cols))
    print("Train images")
    new_col = -1
    for sub in range(1, subjects):
        for ti in range(1, train_imgs+1):
            train_path = './dataset/s{}/{}.pgm'.format(sub, ti)
            img = cv2.imread(train_path, cv2.IMREAD_GRAYSCALE)
            print("{}, {} \n".format(sub, ti))
            print(img)
            print("\n")
      
            img_v = np.array(img, dtype='float64')   #still a mat
            new_col = new_col + 1
            img_v = img_v.flatten()      #img_v is an array now
            #print(img_v.shape)
            img_mat[:, new_col] = img_v
            

    ############## PCA Part ##############
    #img_mat_mean = img_mat.mean(axis=1)
    img_mat_mean = img_mat[:, 0]
    for k in range(1, image_matrix_cols):
        img_mat_mean += img_mat[:, k]
    img_mat_mean /= image_matrix_cols    
    #print(img_mat_mean.shape)
    for k in range(0, image_matrix_cols):
        img_mat1[:, k] = img_mat[:, k] - img_mat_mean
    #img_mat1 = img_mat - img_mat_mean   
    covar_pca = np.dot(img_mat1.T, img_mat1)
    #print(covar_pca.shape)
    covar_pca /= image_matrix_cols       # its a matrix of dim n*n
    egvals_pca, egvecs_pca = np.linalg.eig(covar_pca)           # its a matrix of dim n*n
    egvecs_pca = np.dot(img_mat1, np.matrix(egvecs_pca))
    #egvecs = np.matrix(X1) * np.matrix(egvecs)       # its a matrix of dim d*n
    '''norm = []
    nor = 0
    for vec in egvecs:
        nor = 0
        for j in vec:
            nor += pow(j, 2)
        nor = np.sqrt(nor)  
        norm.append(nor)''' 
    norm = np.linalg.norm(egvecs_pca, axis=0)
    #print("norm shape = {}\n".format(norm.shape))
    egvecs_pca = egvecs_pca / norm
    r1 = image_matrix_cols - subjects
    #print(egvals)
    u = egvals_pca.argsort()[::-1]
    #print(p)
    #print("\n")
    egvecs_pca = egvecs_pca[:, u]

    #print(egvals)
    egvals_pca = egvals_pca[u]
    #print("\n")
    #print(egvals)
    w_pca = egvecs_pca[:,0:r1]
    #w_pca = np.zeros((image_matrix_rows, r1))
 
    
    prj = np.dot(w_pca.T, img_mat1)              # its a matrix of dim m*n

    ############### LDA Part ##############
    mean_total = prj[:, 0]
    for k in range(1, image_matrix_cols):
        mean_total += prj[:, k]
    mean_total /= image_matrix_cols    
    #mean_total = prj.mean(axis=1)
    Sb = np.zeros((r1, r1))
    Sw = np.zeros((r1, r1)) 

    j = 0
    #subjects += 1;
    for i in range(1, subjects):
        prj_i = prj[:, j:j + train_imgs]
        #print(prj_i.shape[1])
        mean_class = prj[:, j]
        for k in range(j+1, j+prj_i.shape[1]):
            mean_class += prj[:, k]
        mean_class /= prj_i.shape[1]
        j += train_imgs
        #print(mean_class)
        Sw += np.matrix(np.matrix(prj_i - mean_class) * np.matrix(prj_i-mean_class).T)
        Sb += image_matrix_cols * np.matrix(np.matrix(mean_class - mean_total) * np.matrix(mean_class - mean_total).T)

    
    #subjects -= 1;
    egvals_lda, egvecs_lda = np.linalg.eig(np.dot(np.linalg.inv(Sw),Sb))
    #performing argsort on egvals
    u = np.argsort(egvals_lda.real)[::-1]
    egvecs_lda = egvecs_lda[:, u]
    #this will sort required egvals
    #selecting u prominent egvecs 
    r2 = subjects - 1
    egvals_lda = egvals_lda[u]
    egvals_lda = np.array((egvals_lda[0:r2]))
    w_fld = egvecs_lda[:, 0:r2]
    #w_fld = egvecs_lda[:, 0:r2].real
    w_opt = np.dot(w_pca, w_fld)
    proj1 = np.matrix(np.matrix(w_opt.T) * np.matrix(img_mat1))
    return proj1, img_mat, w_opt

def test(test_path, proj1, img_mat, w_opt):
    img_read = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)

    img_v = np.array(img_read, dtype='float64')  #img_v is mat
    img_v = img_v.flatten()            #now img_v is an array
    img_v = np.matrix(img_v)
    img_v = img_v.T

    mean = img_mat[:, 0]
    #print(mean.shape)
    for j in range(1, image_matrix_cols):
        mean += img_mat[:, j]
    mean /= image_matrix_cols 
    mean = np.matrix(mean)
    mean = mean.T
    #print(mean.shape)  
    #mean = X.mean(axis=1).reshape(d, 1)
    img_v -= mean
    proj2 = np.dot(w_opt.T, img_v)
    #S = np.dot(w_opt.T, img_v)
    diff = proj1 - proj2
    norms = np.linalg.norm(diff, axis=0)
    #print(norms)
    min_norm = sys.maxsize
    min_norm_index = 0
    #print(min_norm)
    index = 0
    for nom in norms:
        if nom < min_norm:
            min_norm = nom
            min_norm_index = index
        index = index + 1  
    #print(min_norm_index)    
    predicted_id = np.floor((min_norm_index / train_imgs))     
    #print(predicted_id)
    return predicted_id + 1

def main():
    proj1, X, w_opt = train() 
    matched = 0
    for sub in range(1, subjects):
        for ti in range(1, test_imgs+1):
            test_path = './dataset/s{}/{}.pgm'.format(sub, ti+train_imgs)
            predicted_id = test(test_path, proj1, X, w_opt)
            print("Test image from subject {0} matched to subject {1}".format(sub,predicted_id))
            if predicted_id == sub:
                matched += 1
    acc =  (matched*100)/((subjects-1)*test_imgs)        
    print("\nAccuracy = {} %".format(acc)) 

main()
