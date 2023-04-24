import os
import datetime
import cv2
import sqlite3
import numpy as np

db_path = ""


def record(car_id, speed, car_image, plate_image, current_positions, car_img_path, car_plates_path):
    car_image = np.array(car_image)
    img_filename = f'Car_{car_id}_speed_{speed}.jpg'
    img_filepath = os.path.join(car_img_path, img_filename)
    plate_filename = f"Car_plate_{car_id}_speed_{speed}.jpg"
    plate_filepath = os.path.join(car_plates_path, plate_filename)
    box = (current_positions.get(car_id)).astype(int)
    print(box[1], box[3], box[0], box[2])
    if box is not None:
        if not os.path.exists(img_filepath) or not os.path.exists(plate_filepath):
            cv2.imwrite(img_filepath, car_image)
            cv2.imwrite(plate_filepath, plate_image[box[1]:box[3], box[0]:box[2]])
            print("all ok")

            # Convert images to binary data
            with open(img_filepath, 'rb') as f:
                car_img = f.read()
            with open(plate_filepath, 'rb') as f:
                plate_img = f.read()

            # Store the data in the database
            connect = sqlite3.connect(db_path)
            c = connect.cursor()
            c.execute('INSERT INTO Data (car_id, speed, date, Car_image, Plate_image) VALUES (?, ?, ?, ?, ?)',
                      (car_id, speed, datetime.datetime.now().strftime("%m/%d/%Y"), car_img, plate_img))
            connect.commit()
            connect.close()
    else:
        print(f"No bbox found for car {car_id}")
