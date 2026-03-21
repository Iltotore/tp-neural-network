#ifndef __LOAD_MNIST_H
#define __LOAD_MNIST_H

/* lecture de la base mnist : http://yann.lecun.com/exdb/mnist/ */

#define __NB_IMAGES 1
#define __SIZE_IMAGE 28
#define __SIZE_INT 4

typedef struct image {
    unsigned char imgbuf[__SIZE_IMAGE * __SIZE_IMAGE];
    unsigned char label;
} image;

void endianness();
void close_training_files();
void close_test_files();
void open_training_files();
void open_test_files();
void read_training_image(int pos,
                         image* ret);      /*read_one image, at position pos in the training files*/
void read_test_image(int pos, image* ret); /*read_one image, at position pos in the test files*/
void affiche_img(image* rt);

#endif /* __LOAD_MNIST_H*/
