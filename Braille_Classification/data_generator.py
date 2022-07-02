from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import glob

num_list  = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','opening','closing']

datagen = ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.05,
        zoom_range=0.01,
        fill_mode='nearest'
        )


def all_new():
    #alpha = 'a'
    for j in range(len(num_list)):
        path = glob.glob('./images/'+num_list[j]+'/*.png')
        path.sort()
        print(path[2])
        img = load_img(path[2])  # PIL 이미지
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        print(x.shape)

        i = 0
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir='./Braille Dataset2', save_prefix=num_list[j], save_format='jpg'):
            i += 1
            if i > 100:
                break  # 이미지 100장을 생성하고 마칩니다




def single_new(alpha):
    img = load_img('./Braille Dataset/ex/'+alpha+'.png')  # PIL 이미지
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)

    i = 0
    for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='./Braille Dataset/Braille Dataset2', save_prefix=alpha, save_format='jpg'):
        i += 1
        if i > 10:
         break  # 이미지 20장을 생성하고 마칩니다
     
     
all_new()     