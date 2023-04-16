import sqlite3
import requests
import plate
import base64

db_path = ""
url = ''
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("SELECT Car_id, Speed, Date, Car_image, Plate_image FROM Data")
rows = cursor.fetchall()

objects_list = []

for row in rows:
    car_id = row[0]
    speed = row[1]
    date = row[2]
    car_image = row[3]
    plate_image = row[4]
    car_image_b64 = base64.b64encode(car_image).decode("utf-8")
    car_image_b64 = "data:image/png;base64," + car_image_b64
    car_plate = plate.plate_detection(plate_image, 'th')

    obj = {
        'licensePlate': car_plate,
        'date': date,
        'speed': speed,
        'urlimage': car_image_b64
    }

    objects_list.append(obj)

conn.close()
for obj in objects_list:
    x = requests.post(url, json=obj)
    print(x.text)
