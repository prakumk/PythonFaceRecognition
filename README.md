# PythonFaceRecognition
Simple implementation of **Face Recognition** using **python**. It shows how you can train your images and then trained model can be used anywhere for recognition purpose. This example shows how to:

- First process raw images and extract face images.
- Then process those extracted face for training purpose.
- Build trained model from processed face image.
- There are three ways these trained model can be used for recogition purpose, First via ScreenCapture and recognise, Second via Camera device and recognise and Third via creation of Flask API and recognise by uploading image.

# Screenshot
[![Face Recognition Using Python](https://i.ytimg.com/vi/bDY5Q1qbXys/hqdefault.jpg?sqp=-oaymwEZCPYBEIoBSFXyq4qpAwsIARUAAIhCGAFwAQ==&rs=AOn4CLDD4kHeO3IPruQBJP3Ditn98Kqyfg)](https://www.youtube.com/watch?v=bDY5Q1qbXys)

# Link
https://www.youtube.com/watch?v=bDY5Q1qbXys

## Important Library Used :
- Dlib (Dlib is a modern C++ toolkit containing machine learning algorithms and tools for creating complex software in C++ to solve real world problems)

## Code

Extract face from raw images :

```mermaid

detector = dlib.get_frontal_face_detector()

train_dir_name = "raw_images/"
train_output_folder_name = "dataset/"
train_dir = []

for item in os.listdir(train_dir_name):
    if not item.startswith('.'):
        train_dir.append(item)
print(train_dir)

# Loop through each person in the training directory
for person in train_dir:

    # Loop through each training image for the current person
    for person_img in os.listdir(train_dir_name + person):
        if person_img.startswith('.'):
            continue

        print(train_dir_name + person + "/" + person_img)

        number = 0

        image_orig = Image.open(open(train_dir_name + person + "/" + person_img, 'rb'))
        b, g, r = image_orig.split()
        image_orig = Image.merge("RGB", (r, g, b))
        image = np.array(image_orig)

        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # detect faces in the grayscale image
        rects = detector(gray, 1)
        # loop over the face detections
        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            try:
                (x, y, w, h) = face_utils.rect_to_bb(rect)
                # print rect.dtype
                cro = image[y: y + h, x: x + w]

                out_image = cv2.resize(cro, (108, 108))

                folder_name = train_output_folder_name + person

                if os.path.exists(folder_name):
                    pass
                else:
                    os.mkdir(folder_name)

                fram = os.path.join(folder_name + "/", person_img+"_"+str(number) + "." + "jpg")
                number += 1
                cv2.imwrite(fram, out_image)
                # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            except Exception as err:
                print(train_output_folder_name + person + "/", person_img+"_"+str(number) + "." + "jpg")
                print(err)

                pass

```


Train Face to build models :
```

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def train(train_dir_name, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
   
    # The training data would be all the face encodings from all the known images and the labels are their names
    encodings = []
    names = []

    # Training directory
    train_dir = []

    for item in os.listdir(train_dir_name):
        if not item.startswith('.'):
            train_dir.append(item)
    print(train_dir)

    # Loop through each person in the training directory
    for person in train_dir:

        # Loop through each training image for the current person
        for person_img in os.listdir(train_dir_name + person):
            if person_img.startswith('.'):
                continue
            # Get the face encodings for the face in each image file
            print(train_dir_name + person + "/" + person_img)
            face = face_recognition.load_image_file(train_dir_name + person + "/" + person_img)
            # print(face.shape)
            # Assume the whole image is the location of the face
            #height, width, _ = face.shape
            # location is in css order - top, right, bottom, left
            height, width, _ = face.shape
            face_location = (0, width, height, 0)
            # print(width,height)

            face_enc = face_recognition.face_encodings(face, known_face_locations=[face_location])
            
            

            face_enc = np.array(face_enc)
            face_enc = face_enc.flatten()
        
            # Add face encoding for current image with corresponding label (name) to the training data
            encodings.append(face_enc)
            names.append(person)

    # print(np.array(encodings).shape)
    # print(np.array(names).shape)
    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(encodings,names)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf



if __name__ == "__main__":
    classifier = train("dataset/", model_save_path="model/trained_model.dat", n_neighbors=2)
    print("Training complete!")    
```

