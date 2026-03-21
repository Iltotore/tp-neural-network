#include "load_mnist.h"

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>

/* eurk, globals */
int fd1, fd2, fd3, fd4;

/*----------little to big endian ----------------------*/
void endianness() {
    int x = 1;

    char* y = (char*)&x;

    if (*y + 48 == '0') printf("big endian\n");
    // else printf("little endian\n");
}

//! Byte swap int
int32_t swap_int32(int32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | ((val >> 16) & 0xFFFF);
}

void close_training_files() {
    close(fd1);
    close(fd2);
}

void close_test_files() {
    close(fd3);
    close(fd4);
}

/*------------handling MNIST Files-----------------------*/

void open_training_files() {
    int magic_label, magic_train;       /*magic numbers*/
    int nb_items;                       /*label set*/
    int nb_images, nb_rows, nb_columns; /* number of images of the training set 60000*/
    int size_image;                     /*image size, should be 28*/
    int temp, lu;

    fd1 = open("./train-images-idx3-ubyte", O_RDONLY);
    fd2 = open("./train-labels-idx1-ubyte", O_RDONLY);

    int swaped;

    read(fd1, &temp, sizeof(int));
    magic_train = swap_int32(temp);
    read(fd1, &temp, sizeof(int));
    nb_items = swap_int32(temp);
    read(fd1, &temp, sizeof(int));
    nb_rows = swap_int32(temp);
    lu = read(fd1, &temp, sizeof(int));
    nb_columns = swap_int32(temp);

    /* training set */
    //  printf("magic number : %d\n",magic_train);
    //  printf("nb images : %d\n",nb_items);
    //  printf("nb rows : %d\n",nb_rows);
    //  printf("lu : %d nb columns : %d\n",lu, nb_columns);
}

void open_test_files() {
    int magic_label, magic_test;        /*magic numbers*/
    int nb_items;                       /*label set*/
    int nb_images, nb_rows, nb_columns; /* number of images of the training set 60000*/
    int size_image;                     /*image size, should be 28*/
    int temp, lu;

    fd3 = open("./t10k-images-idx3-ubyte", O_RDONLY);
    fd4 = open("./t10k-labels-idx1-ubyte", O_RDONLY);

    int swaped;

    read(fd3, &temp, sizeof(int));
    magic_test = swap_int32(temp);
    read(fd3, &temp, sizeof(int));
    nb_items = swap_int32(temp);
    read(fd3, &temp, sizeof(int));
    nb_rows = swap_int32(temp);
    lu = read(fd3, &temp, sizeof(int));
    nb_columns = swap_int32(temp);

    /* training set */
    //  printf("magic number : %d\n",magic_train);
    //  printf("nb images : %d\n",nb_items);
    //  printf("nb rows : %d\n",nb_rows);
    //  printf("lu : %d nb columns : %d\n",lu, nb_columns);
}

void read_training_image(int pos,
                         image* ret) { /*read_one image, at position pos in the training files*/
    int i;
    long int offset;

    /*for the training set image file*/

    offset = 4 * 4; /*the file header : 4 32bits integers*/

    offset +=
        __SIZE_IMAGE * __SIZE_IMAGE *
        pos; /* one MNIST image is 28x28 bytes, we have to skip pos images before the good one */
    pread(fd1, ret->imgbuf, __SIZE_IMAGE * __SIZE_IMAGE, offset); /*read the image*/

    /*for the training set label file*/
    offset = 2 * 4;
    offset += pos; /* one MNIST label is one bytes, we have to skip pos bytes before the good one */
    pread(fd2, &(ret->label), 1, offset); /*read the image*/
}

void read_test_image(int pos, image* ret) { /*read_one image, at position pos in the test files*/
    int i;
    long int offset;

    /*for the training set image file*/

    offset = 4 * 4; /*the file header : 4 32bits integers*/

    offset +=
        __SIZE_IMAGE * __SIZE_IMAGE *
        pos; /* one MNIST image is 28x28 bytes, we have to skip pos images before the good one */
    pread(fd3, ret->imgbuf, __SIZE_IMAGE * __SIZE_IMAGE, offset); /*read the image*/

    /*for the test set label file*/
    offset = 2 * 4;
    offset += pos; /* one MNIST label is one bytes, we have to skip pos bytes before the good one */
    pread(fd4, &(ret->label), 1, offset); /*read the image*/
}

void affiche_img(image* rt) {
    int i;
    int chargl = '#';

    for (i = 0; i < __SIZE_IMAGE * __SIZE_IMAGE; i++) {
        if (i % __SIZE_IMAGE == 0) fprintf(stderr, "\n");

        if (rt->imgbuf[i] < 200) chargl = '*';
        if (rt->imgbuf[i] < 150) chargl = '+';
        if (rt->imgbuf[i] < 100) chargl = '=';
        if (rt->imgbuf[i] < 50) chargl = '-';
        if (rt->imgbuf[i] < 25) chargl = ':';
        if (rt->imgbuf[i] < 5) chargl = '.';
        if (rt->imgbuf[i] == 0) chargl = ' ';
        fprintf(stderr, "%c", chargl);
    }
    fprintf(stderr, "\n");
    fprintf(stderr, "-----------label : %d-------\n", rt->label);
}
